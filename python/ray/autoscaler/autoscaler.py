from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import hashlib
import json
import logging
import math
import os
import subprocess
import threading
import time
from collections import defaultdict

import numpy as np
from scipy.optimize import linprog
import ray.services as services
import yaml
from ray.services import create_redis_client
from ray.worker import global_worker
from ray.autoscaler.docker import dockerize_if_needed
from ray.autoscaler.node_provider import get_node_provider, \
    get_default_config
from ray.autoscaler.tags import (TAG_RAY_LAUNCH_CONFIG, TAG_RAY_RUNTIME_CONFIG,
                                 TAG_RAY_NODE_STATUS, TAG_RAY_NODE_TYPE,
                                 TAG_RAY_NODE_NAME)
from ray.autoscaler.updater import NodeUpdaterThread
from ray.ray_constants import AUTOSCALER_MAX_NUM_FAILURES, \
    AUTOSCALER_MAX_LAUNCH_BATCH, AUTOSCALER_MAX_CONCURRENT_LAUNCHES, \
    AUTOSCALER_UPDATE_INTERVAL_S, AUTOSCALER_HEARTBEAT_TIMEOUT_S, \
    AUTOSCALER_RESOURCE_REQUEST_CHANNEL, MEMORY_RESOURCE_UNIT_BYTES, \
    AUTOSCALER_STATUS_CHANNEL
from six import string_types
from six.moves import queue

logger = logging.getLogger(__name__)

REQUIRED, OPTIONAL = True, False

# For (a, b), if a is a dictionary object, then
# no extra fields can be introduced.

CLUSTER_CONFIG_SCHEMA = {
    # An unique identifier for the head node and workers of this cluster.
    "cluster_name": (str, REQUIRED),

    # The minimum number of workers nodes to launch in addition to the head
    # node. This number should be >= 0.
    "min_workers": (int, OPTIONAL),

    # The maximum number of workers nodes to launch in addition to the head
    # node. This takes precedence over min_workers.
    "max_workers": (int, REQUIRED),

    # The number of workers to launch initially, in addition to the head node.
    "initial_workers": (int, OPTIONAL),

    # The mode of the autoscaler e.g. default, aggressive
    "autoscaling_mode": (str, OPTIONAL),

    # The autoscaler will scale up the cluster to this target fraction of
    # resources usage. For example, if a cluster of 8 nodes is 100% busy
    # and target_utilization was 0.8, it would resize the cluster to 10.
    "target_utilization_fraction": (float, OPTIONAL),

    # If a node is idle for this many minutes, it will be removed.
    "idle_timeout_minutes": (int, OPTIONAL),

    # Cloud-provider specific configuration.
    "provider": (
        {
            "type": (str, REQUIRED),  # e.g. aws
            "region": (str, OPTIONAL),  # e.g. us-east-1
            "availability_zone": (str, OPTIONAL),  # e.g. us-east-1a
            "module": (str,
                       OPTIONAL),  # module, if using external node provider
            "project_id": (None, OPTIONAL),  # gcp project id, if using gcp
            "head_ip": (str, OPTIONAL),  # local cluster head node
            "worker_ips": (list, OPTIONAL),  # local cluster worker nodes
            "use_internal_ips": (bool, OPTIONAL),  # don't require public ips
            "extra_config": (dict, OPTIONAL),  # provider-specific config
        },
        REQUIRED),

    # How Ray will authenticate with newly launched nodes.
    "auth": (
        {
            "ssh_user": (str, REQUIRED),  # e.g. ubuntu
            "ssh_private_key": (str, OPTIONAL),
        },
        REQUIRED),

    # Docker configuration. If this is specified, all setup and start commands
    # will be executed in the container.
    "docker": (
        {
            "image": (str, OPTIONAL),  # e.g. tensorflow/tensorflow:1.5.0-py3
            "container_name": (str, OPTIONAL),  # e.g., ray_docker
            # shared options for starting head/worker docker
            "run_options": (list, OPTIONAL),

            # image for head node, takes precedence over "image" if specified
            "head_image": (str, OPTIONAL),
            # head specific run options, appended to run_options
            "head_run_options": (list, OPTIONAL),
            # analogous to head_image
            "worker_image": (str, OPTIONAL),
            # analogous to head_run_options
            "worker_run_options": (list, OPTIONAL),
        },
        OPTIONAL),

    # Provider-specific config for the head node, e.g. instance type.
    "head_node": (dict, OPTIONAL),

    # Provider-specific config for worker nodes. e.g. instance type.
    "worker_nodes": (dict, OPTIONAL),

    # Map of remote paths to local paths, e.g. {"/tmp/data": "/my/local/data"}
    "file_mounts": (dict, OPTIONAL),

    # List of commands that will be run before `setup_commands`. If docker is
    # enabled, these commands will run outside the container and before docker
    # is setup.
    "initialization_commands": (list, OPTIONAL),

    # List of common shell commands to run to setup nodes.
    "setup_commands": (list, OPTIONAL),

    # Commands that will be run on the head node after common setup.
    "head_setup_commands": (list, OPTIONAL),

    # Commands that will be run on worker nodes after common setup.
    "worker_setup_commands": (list, OPTIONAL),

    # Command to start ray on the head node. You shouldn't need to modify this.
    "head_start_ray_commands": (list, OPTIONAL),

    # Command to start ray on worker nodes. You shouldn't need to modify this.
    "worker_start_ray_commands": (list, OPTIONAL),

    # Whether to avoid restarting the cluster during updates. This field is
    # controlled by the ray --no-restart flag and cannot be set by the user.
    "no_restart": (None, OPTIONAL),
}


class LoadMetrics(object):
    """Container for cluster load metrics.

    Metrics here are updated from raylet heartbeats. The autoscaler
    queries these metrics to determine when to scale up, and which nodes
    can be removed.
    """

    def __init__(self):
        self.last_used_time_by_ip = {}
        self.last_heartbeat_time_by_ip = {}
        self.static_resources_by_ip = {}
        self.dynamic_resources_by_ip = {}
        self.resource_load_by_ip = {}
        self.local_ip = services.get_node_ip_address()

    def update(self, ip, static_resources, dynamic_resources, resource_load):
        self.resource_load_by_ip[ip] = resource_load
        self.static_resources_by_ip[ip] = static_resources

        # We are not guaranteed to have a corresponding dynamic resource for
        # every static resource because dynamic resources are based on the
        # available resources in the heartbeat, which does not exist if it is
        # zero. Thus, we have to update dynamic resources here.
        dynamic_resources_update = dynamic_resources.copy()
        for resource_name, capacity in static_resources.items():
            if resource_name not in dynamic_resources_update:
                dynamic_resources_update[resource_name] = 0.0
        self.dynamic_resources_by_ip[ip] = dynamic_resources_update

        now = time.time()
        if ip not in self.last_used_time_by_ip or \
                static_resources != dynamic_resources:
            self.last_used_time_by_ip[ip] = now
        self.last_heartbeat_time_by_ip[ip] = now

    def mark_active(self, ip):
        assert ip is not None, "IP should be known at this time"
        logger.info("Node {} is newly setup, treating as active".format(ip))
        self.last_heartbeat_time_by_ip[ip] = time.time()

    def prune_active_ips(self, active_ips):
        active_ips = set(active_ips)
        active_ips.add(self.local_ip)

        def prune(mapping):
            unwanted = set(mapping) - active_ips
            for unwanted_key in unwanted:
                logger.info("LoadMetrics: "
                            "Removed mapping: {} - {}".format(
                                unwanted_key, mapping[unwanted_key]))
                del mapping[unwanted_key]
            if unwanted:
                logger.info(
                    "LoadMetrics: "
                    "Removed {} stale ip mappings: {} not in {}".format(
                        len(unwanted), unwanted, active_ips))
            assert not (unwanted & set(mapping))

        prune(self.last_used_time_by_ip)
        prune(self.static_resources_by_ip)
        prune(self.dynamic_resources_by_ip)
        prune(self.resource_load_by_ip)
        prune(self.last_heartbeat_time_by_ip)

    def approx_workers_used(self):
        return self._info()["NumNodesUsed"]

    def num_workers_connected(self):
        return self._info()["NumNodesConnected"]

    def get_resource_usage(self):
        num_nodes = len(self.static_resources_by_ip)
        nodes_used = 0.0
        num_nonidle = 0
        has_saturated_node = False
        resources_used = {}
        resources_total = {}
        for ip, max_resources in self.static_resources_by_ip.items():
            avail_resources = self.dynamic_resources_by_ip[ip]
            resource_load = self.resource_load_by_ip[ip]
            max_frac = 0.0
            for resource_id, amount in resource_load.items():
                if amount > 0:
                    has_saturated_node = True
                    max_frac = 1.0  # the resource is saturated
            for resource_id, amount in max_resources.items():
                used = amount - avail_resources[resource_id]
                if resource_id not in resources_used:
                    resources_used[resource_id] = 0.0
                    resources_total[resource_id] = 0.0
                resources_used[resource_id] += used
                resources_total[resource_id] += amount
                used = max(0, used)
                if amount > 0:
                    frac = used / float(amount)
                    if frac > max_frac:
                        max_frac = frac
            nodes_used += max_frac
            if max_frac > 0:
                num_nonidle += 1

        # If any nodes have a queue buildup, assume all non-idle nodes are 100%
        # busy, plus the head node. This guards against the case of not scaling
        # up due to poor task packing.
        if has_saturated_node:
            nodes_used = min(num_nonidle + 1.0, num_nodes)

        return nodes_used, resources_used, resources_total

    def info_string(self):
        return ", ".join(
            ["{}={}".format(k, v) for k, v in sorted(self._info().items())])

    def _info(self):
        nodes_used, resources_used, resources_total = self.get_resource_usage()

        now = time.time()
        idle_times = [now - t for t in self.last_used_time_by_ip.values()]
        heartbeat_times = [
            now - t for t in self.last_heartbeat_time_by_ip.values()
        ]
        most_delayed_heartbeats = sorted(
            list(self.last_heartbeat_time_by_ip.items()),
            key=lambda pair: pair[1])[:5]
        most_delayed_heartbeats = {
            ip: (now - t)
            for ip, t in most_delayed_heartbeats
        }

        def format_resource(key, value):
            if key in ["object_store_memory", "memory"]:
                return "{} GiB".format(
                    round(value * MEMORY_RESOURCE_UNIT_BYTES / 1e9, 2))
            else:
                return round(value, 2)

        return {
            "ResourceUsage": ", ".join([
                "{}/{} {}".format(
                    format_resource(rid, resources_used[rid]),
                    format_resource(rid, resources_total[rid]), rid)
                for rid in sorted(resources_used)
            ]),
            "NumNodesConnected": len(self.static_resources_by_ip),
            "NumNodesUsed": round(nodes_used, 2),
            "NodeIdleSeconds": "Min={} Mean={} Max={}".format(
                int(np.min(idle_times)) if idle_times else -1,
                int(np.mean(idle_times)) if idle_times else -1,
                int(np.max(idle_times)) if idle_times else -1),
            "TimeSinceLastHeartbeat": "Min={} Mean={} Max={}".format(
                int(np.min(heartbeat_times)) if heartbeat_times else -1,
                int(np.mean(heartbeat_times)) if heartbeat_times else -1,
                int(np.max(heartbeat_times)) if heartbeat_times else -1),
            "MostDelayedHeartbeats": most_delayed_heartbeats,
        }


class NodeLauncher(threading.Thread):
    def __init__(self, provider, queue, pending, index=None, *args, **kwargs):
        self.queue = queue
        self.pending = pending
        self.provider = provider
        self.index = str(index) if index is not None else ""
        super(NodeLauncher, self).__init__(*args, **kwargs)

    def _launch_node(self, config, count):
        worker_filter = {TAG_RAY_NODE_TYPE: "worker"}
        before = self.provider.non_terminated_nodes(tag_filters=worker_filter)
        launch_hash = hash_launch_conf(config["worker_nodes"], config["auth"])
        self.log("Launching {} nodes.".format(count))
        self.provider.create_node(
            config["worker_nodes"], {
                TAG_RAY_NODE_NAME: "ray-{}-worker".format(
                    config["cluster_name"]),
                TAG_RAY_NODE_TYPE: "worker",
                TAG_RAY_NODE_STATUS: "uninitialized",
                TAG_RAY_LAUNCH_CONFIG: launch_hash,
            }, count)
        after = self.provider.non_terminated_nodes(tag_filters=worker_filter)
        if set(after).issubset(before):
            self.log("No new nodes reported after node creation.")

    def run(self):
        while True:
            config, count = self.queue.get()
            self.log("Got {} nodes to launch.".format(count))
            try:
                self._launch_node(config, count)
            except Exception:
                logger.exception("Launch failed")
            finally:
                self.pending.dec(count)

    def log(self, statement):
        prefix = "NodeLauncher{}:".format(self.index)
        logger.info(prefix + " {}".format(statement))


class ConcurrentCounter():
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, count):
        with self._lock:
            self._value += count
            return self._value

    def dec(self, count):
        with self._lock:
            assert self._value >= count, "counter cannot go negative"
            self._value -= count
            return self._value

    @property
    def value(self):
        with self._lock:
            return self._value


class ManualAutoscaler(object):
    """
    A manual implementation of the StandardAutoscaler that only fulfils explicit resource requests.

    It supports multiple resource types and multiple worker types, each with a potentially different list of resources
    associated with it.
    """

    def __init__(self,
                 config_path,
                 load_metrics,
                 max_launch_batch=AUTOSCALER_MAX_LAUNCH_BATCH,
                 max_concurrent_launches=AUTOSCALER_MAX_CONCURRENT_LAUNCHES,
                 max_failures=AUTOSCALER_MAX_NUM_FAILURES,
                 process_runner=subprocess,
                 update_interval_s=AUTOSCALER_UPDATE_INTERVAL_S,
                 redis_address=None,
                 redis_password=None):

        self.load_metrics = None  # NOTE ignore for now
        self.process_runner = process_runner

        # Config.
        self.config_path = config_path
        self.reload_config(errors_fatal=True)

        # Redis.
        self.redis_address = redis_address
        self.redis_password = redis_password
        self.redis_client = create_redis_client(self.redis_address, password=self.redis_password)

        # Map from node_id to NodeUpdater processes
        self.updaters = {}
        self.num_failed_updates = defaultdict(int)
        self.num_successful_updates = defaultdict(int)
        self.num_failures = 0
        self.last_update_time = 0.0
        self.update_interval_s = update_interval_s
        self.bringup = True

        # Launch and provision.
        self.provider = get_node_provider(self.config["provider"], self.config["cluster_name"])
        self.max_failures = max_failres
        self.max_launch_batch = max_launch_batch
        self.max_concurrent_launches = max_concurrent_launches

        self.launch_queue = queue.Queue()
        self.num_launches_pending = ConcurrentCounter()
        max_batches = math.ceil(max_concurrent_launches / float(max_launch_batch))
        for i in range(int(max_batches)):
            node_launcher = NodeLauncher(provider=self.provider, queue=self.launch_queue,
                                         index=i, pending=self.num_launches_pending)
            node_launcher.daemon = True
            node_launcher.start()

        # Expand local file_mounts to allow ~ in the paths. This can't be done
        # earlier when the config is written since we might be on different
        # platform and the expansion would result in wrong path.
        self.config["file_mounts"] = {
            remote: os.path.expanduser(local)
            for remote, local in self.config["file_mounts"].items()
        }

        for local_path in self.config["file_mounts"].values():
            assert os.path.exists(local_path)

        self.resource_requests = defaultdict(int)

        logger.info("ManualAutoscaler: {}".format(self.config))

    def reload_config(self, errors_fatal=False):
        try:
            with open(self.config_path) as f:
                new_config = yaml.safe_load(f)
            validate_config(new_config)
            self.config = new_config

            # We now have multiple valid launch hashes.
            self.launch_hashes = {hash_launch_conf(worker_config, new_config["auth"]): worker_name
                                  for worker_name, worker_config in new_config["worker_nodes"].items()}

            # Don't include worker_start_ray_commands in the runtime hash, since we will want to dynamically update it.
            self.runtime_hash = hash_runtime_conf(new_config["file_mounts"], [
                                                  new_config["worker_setup_commands"]])
        except Exception as e:
            if errors_fatal:
                raise e
            else:
                logger.exception("ManualAutoscaler: "
                                 "Errors parsing config.")

    def workers(self):
        return self.provider.non_terminated_nodes(
            tag_filters={TAG_RAY_NODE_TYPE: "worker"})

    def request_resources(self, resources):
        self.resource_requests = copy.deepcopy(resources)
        logger.info("ManualAutoscaler: resource_requests={}".format(self.resource_requests))

    def optimize_requirements(self):

        if not self.resource_requests:
            # Keep everything the same.
            return None

        # Populate worker resources, injecting one unit of worker name to each worker type, and worker costs.
        worker_resources = {}
        worker_costs = {}
        for worker_name, worker_config in self.config["worker_nodes"].items():
            worker_resources[worker_name] = worker_config["Resources"].copy()
            worker_resources[worker_name].update({worker_name: 1})
            worker_costs[worker_name] = worker_config["Cost"]
        worker_resources = {worker_name: worker_config["Resources"].copy()
                            for worker_name, worker_config in self.config["worker_nodes"]}

        new_requests = copy.deepcopy(self.resource_requests)
        self.resource_requests = {}  # reset

        # Arrayize
        available_resource_types = set()
        for worker_name, resources in worker_resources.items():
            for resource_name, _ in resources.items():
                available_resource_types.add(resource_name)

        requested_resource_types = set(new_requests.keys())

        assert requested_resource_types.issuubset(available_resource_types)

        sorted_workers = sorted(worker_resources.keys())
        sorted_resources = sorted(available_resource_types)

        cost = np.array([worker_costs[worker_name] for worker_name in sorted_workers])
        resources_mat = np.array([[worker_resources[worker_name].get(resource_name, 0) for resource_name in sorted_resources]
                                  for worker_name in sorted_workers]).T
        request = np.array([new_requests.get(resource_type, 0) for resource_type in sorted_resources])

        # Optimize.
        # TODO ILP. This is silly.
        solution = linprog(cost, -resources_mat, -request)
        # Dictionarize.
        worker_requirements = {sorted_workers[i]: int(np.ceil(solution['x'][i])) for i in range(len(sorted_workers))}

        return worker_requirements

    def update(self):
        try:
            self.reload_config(errors_fatal=False)
            self._update()
        except Exception as e:
            logger.exception("ManualAutoscaler: "
                             "Error during autoscaling.")
            self.num_failures += 1
            if self.num_failures > self.max_failures:
                logger.critical("ManualAutoscaler: "
                                "Too many errors, abort.")
                raise e

    def _update(self):

        now = time.time()

        # Throttle autoscaling updates to this interval to avoid exceeding
        # rate limits on API calls.
        if now - self.last_update_time < self.update_interval_s:
            return

        self.last_update_time = now
        num_pending = self.num_launches_pending.value
        nodes = self.workers()

        # Determine new requirements.
        worker_requirements = self.optimize_requirements()

        # Figure out the delta from the current requirements.
        if not self.current_requirements:
            delta_requirements = worker_requirements
        else:
            delta_requirements = {worker_name: worker_requirements[worker_name] - self.current_requirements[worker_name]
                                  for worker_name in self.config["worker_nodes"].keys()}


        for worker_name, worker_count in delta_requirements.items():
            if worker_count < 0:
                # Take down obsolete nodes.
                nodes = self.workers()

            elif worker_count > 0:
                # Spin up new nodes.
                self.launch_new_node(worker_count, worker_name)

        # Process any completed updates
        completed = []
        for node_id, updater in self.updaters.items():
            if not updater.is_alive():
                completed.append(node_id)
        if completed:
            for node_id in completed:
                if self.updaters[node_id].exitcode == 0:
                    self.num_successful_updates[node_id] += 1
                else:
                    self.num_failed_updates[node_id] += 1
                del self.updaters[node_id]
            # Mark the node as active to prevent the node recovery logic
            # immediately trying to restart Ray on the new node.
            self.load_metrics.mark_active(self.provider.internal_ip(node_id))
            nodes = self.workers()
            self.log_info_string(nodes, target_workers)

        # Update nodes with out-of-date files
        T = [
            threading.Thread(
                target=self.spawn_updater,
                args=(node_id, commands),
            ) for node_id, commands in (self.should_update(node_id)
                                        for node_id in nodes)
            if node_id is not None
        ]
        for t in T:
            t.start()
        for t in T:
            t.join()

        # Attempt to recover unhealthy nodes
        for node_id in nodes:
            self.recover_if_needed(node_id, now)

        self.publish_autoscaler_status(nodes, target_workers)


class StandardAutoscaler(object):
    """The autoscaling control loop for a Ray cluster.

    There are two ways to start an autoscaling cluster: manually by running
    `ray start --head --autoscaling-config=/path/to/config.yaml` on a
    instance that has permission to launch other instances, or you can also use
    `ray create_or_update /path/to/config.yaml` from your laptop, which will
    configure the right AWS/Cloud roles automatically.

    StandardAutoscaler's `update` method is periodically called by `monitor.py`
    to add and remove nodes as necessary. Currently, load-based autoscaling is
    not implemented, so all this class does is try to maintain a constant
    cluster size.

    StandardAutoscaler is also used to bootstrap clusters (by adding workers
    until the target cluster size is met).
    """

    def __init__(self,
                 config_path,
                 load_metrics,
                 max_launch_batch=AUTOSCALER_MAX_LAUNCH_BATCH,
                 max_concurrent_launches=AUTOSCALER_MAX_CONCURRENT_LAUNCHES,
                 max_failures=AUTOSCALER_MAX_NUM_FAILURES,
                 process_runner=subprocess,
                 update_interval_s=AUTOSCALER_UPDATE_INTERVAL_S,
                 redis_address=None,
                 redis_password=None):
        self.config_path = config_path
        self.reload_config(errors_fatal=True)
        self.load_metrics = load_metrics
        self.provider = get_node_provider(self.config["provider"],
                                          self.config["cluster_name"])

        self.max_failures = max_failures
        self.max_launch_batch = max_launch_batch
        self.max_concurrent_launches = max_concurrent_launches
        self.process_runner = process_runner
        self.redis_address = redis_address
        self.redis_password = redis_password
        self.redis_client = create_redis_client(self.redis_address, password=self.redis_password)
        self.redis_key = AUTOSCALER_STATUS_CHANNEL

        # Map from node_id to NodeUpdater processes
        self.updaters = {}
        self.num_failed_updates = defaultdict(int)
        self.num_successful_updates = defaultdict(int)
        self.num_failures = 0
        self.last_update_time = 0.0
        self.update_interval_s = update_interval_s
        self.bringup = True

        # Node launchers
        self.launch_queue = queue.Queue()
        self.num_launches_pending = ConcurrentCounter()
        max_batches = math.ceil(
            max_concurrent_launches / float(max_launch_batch))
        for i in range(int(max_batches)):
            node_launcher = NodeLauncher(
                provider=self.provider,
                queue=self.launch_queue,
                index=i,
                pending=self.num_launches_pending)
            node_launcher.daemon = True
            node_launcher.start()

        # Expand local file_mounts to allow ~ in the paths. This can't be done
        # earlier when the config is written since we might be on different
        # platform and the expansion would result in wrong path.
        self.config["file_mounts"] = {
            remote: os.path.expanduser(local)
            for remote, local in self.config["file_mounts"].items()
        }

        for local_path in self.config["file_mounts"].values():
            assert os.path.exists(local_path)

        self.resource_requests = defaultdict(int)

        logger.info("StandardAutoscaler: {}".format(self.config))

    def update(self):
        try:
            self.reload_config(errors_fatal=False)
            self._update()
        except Exception as e:
            logger.exception("StandardAutoscaler: "
                             "Error during autoscaling.")
            self.num_failures += 1
            if self.num_failures > self.max_failures:
                logger.critical("StandardAutoscaler: "
                                "Too many errors, abort.")
                raise e

    def _update(self):
        now = time.time()

        # Throttle autoscaling updates to this interval to avoid exceeding
        # rate limits on API calls.
        if now - self.last_update_time < self.update_interval_s:
            return

        self.last_update_time = now
        num_pending = self.num_launches_pending.value
        nodes = self.workers()
        self.load_metrics.prune_active_ips(
            [self.provider.internal_ip(node_id) for node_id in nodes])
        target_workers, worker_name = self.target_num_workers()

        if len(nodes) >= target_workers:
            if "CPU" in self.resource_requests:
                del self.resource_requests["CPU"]

        self.log_info_string(nodes, target_workers)

        # Terminate any idle or out of date nodes
        last_used = self.load_metrics.last_used_time_by_ip
        horizon = now - (60 * self.config["idle_timeout_minutes"])

        nodes_to_terminate = []
        for node_id in nodes:
            node_ip = self.provider.internal_ip(node_id)
            if node_ip in last_used and last_used[node_ip] < horizon and \
                    len(nodes) - len(nodes_to_terminate) > target_workers:
                logger.info("StandardAutoscaler: "
                            "{}: Terminating idle node".format(node_id))
                nodes_to_terminate.append(node_id)
            elif not self.launch_config_ok(node_id):
                logger.info("StandardAutoscaler: "
                            "{}: Terminating outdated node".format(node_id))
                nodes_to_terminate.append(node_id)

        if nodes_to_terminate:
            self.provider.terminate_nodes(nodes_to_terminate)
            nodes = self.workers()
            self.log_info_string(nodes, target_workers)

        # Terminate nodes if there are too many
        nodes_to_terminate = []
        while len(nodes) > self.config["max_workers"]:
            logger.info("StandardAutoscaler: "
                        "{}: Terminating unneeded node".format(nodes[-1]))
            nodes_to_terminate.append(nodes[-1])
            nodes = nodes[:-1]

        if nodes_to_terminate:
            self.provider.terminate_nodes(nodes_to_terminate)
            nodes = self.workers()
            self.log_info_string(nodes, target_workers)

        # Launch new nodes if needed
        num_workers = len(nodes) + num_pending
        if num_workers < target_workers:
            max_allowed = min(self.max_launch_batch,
                              self.max_concurrent_launches - num_pending)

            num_launches = min(max_allowed, target_workers - num_workers)
            self.launch_new_node(num_launches, worker_name)
            nodes = self.workers()
            self.log_info_string(nodes, target_workers)
        elif self.load_metrics.num_workers_connected() >= target_workers:
            logger.info("Ending bringup phase")
            self.bringup = False
            self.log_info_string(nodes, target_workers)

        # Process any completed updates
        completed = []
        for node_id, updater in self.updaters.items():
            if not updater.is_alive():
                completed.append(node_id)
        if completed:
            for node_id in completed:
                if self.updaters[node_id].exitcode == 0:
                    self.num_successful_updates[node_id] += 1
                else:
                    self.num_failed_updates[node_id] += 1
                del self.updaters[node_id]
            # Mark the node as active to prevent the node recovery logic
            # immediately trying to restart Ray on the new node.
            self.load_metrics.mark_active(self.provider.internal_ip(node_id))
            nodes = self.workers()
            self.log_info_string(nodes, target_workers)

        # Update nodes with out-of-date files
        T = [
            threading.Thread(
                target=self.spawn_updater,
                args=(node_id, commands),
            ) for node_id, commands in (self.should_update(node_id)
                                        for node_id in nodes)
            if node_id is not None
        ]
        for t in T:
            t.start()
        for t in T:
            t.join()

        # Attempt to recover unhealthy nodes
        for node_id in nodes:
            self.recover_if_needed(node_id, now)

        self.publish_autoscaler_status(nodes, target_workers)

    def reload_config(self, errors_fatal=False):
        try:
            with open(self.config_path) as f:
                new_config = yaml.safe_load(f.read())
            validate_config(new_config)
            # Inject worker names as resources.
            for worker_name, worker_config in new_config["worker_nodes"].items():
                worker_config["Resources"]["worker_type"] = worker_name
            #new_launch_hash = hash_launch_conf(new_config["worker_nodes"],
            #                                   new_config["auth"])
            new_launch_hashes = {hash_launch_conf(worker_config, new_config["auth"]): worker_name
                                 for worker_name, worker_config in new_config["worker_nodes"].items()}
            new_runtime_hash = hash_runtime_conf(new_config["file_mounts"], [
                new_config["worker_setup_commands"],
                new_config["worker_start_ray_commands"]
            ])
            self.config = new_config
            #self.launch_hash = new_launch_hash
            self.launch_hashes = new_launch_hashes
            self.runtime_hash = new_runtime_hash
        except Exception as e:
            if errors_fatal:
                raise e
            else:
                logger.exception("StandardAutoscaler: "
                                 "Error parsing config.")

    def target_num_workers(self):

        # TODO optimization problem (integer linear programming, probably)
        # TODO for now, we can accept an explicit worker type request
        worker_name = self.resource_requests.pop("worker_type", sorted(self.config["worker_nodes"])[0])

        target_frac = self.config["target_utilization_fraction"]
        cur_used = self.load_metrics.approx_workers_used()
        ideal_num_nodes = int(np.ceil(cur_used / float(target_frac)))
        ideal_num_workers = ideal_num_nodes - 1  # subtract 1 for head node

        initial_workers = self.config["initial_workers"]
        aggressive = self.config["autoscaling_mode"] == "aggressive"
        if self.bringup:
            ideal_num_workers = max(ideal_num_workers, initial_workers)
        elif aggressive and cur_used > 0:
            # If we want any workers, we want at least initial_workers
            ideal_num_workers = max(ideal_num_workers, initial_workers)

        # Other resources are not supported at present.
        if "CPU" in self.resource_requests:
            try:
                #cores_per_worker = self.config["worker_nodes"]["Resources"]["CPU"]
                cores_per_worker = self.config["worker_nodes"][worker_name]["Resources"]["CPU"]
            except KeyError:
                cores_per_worker = 1  # Assume the worst

            cores_desired = self.resource_requests.pop("CPU")

            ideal_num_workers = max(
                ideal_num_workers,
                int(np.ceil(cores_desired / cores_per_worker)))

        return min(self.config["max_workers"],
                   max(self.config["min_workers"], ideal_num_workers)), worker_name

    def launch_config_ok(self, node_id):
        launch_conf = self.provider.node_tags(node_id).get(
            TAG_RAY_LAUNCH_CONFIG)
        #if self.launch_hash != launch_conf:
        if launch_conf not in self.launch_hashes:
            return False
        return True

    def files_up_to_date(self, node_id):
        applied = self.provider.node_tags(node_id).get(TAG_RAY_RUNTIME_CONFIG)
        if applied != self.runtime_hash:
            logger.info("StandardAutoscaler: "
                        "{}: Runtime state is {}, want {}".format(
                            node_id, applied, self.runtime_hash))
            return False
        return True

    def recover_if_needed(self, node_id, now):
        if not self.can_update(node_id):
            return
        key = self.provider.internal_ip(node_id)
        if key not in self.load_metrics.last_heartbeat_time_by_ip:
            self.load_metrics.last_heartbeat_time_by_ip[key] = now
        last_heartbeat_time = self.load_metrics.last_heartbeat_time_by_ip[key]
        delta = now - last_heartbeat_time
        if delta < AUTOSCALER_HEARTBEAT_TIMEOUT_S:
            return
        logger.warning("StandardAutoscaler: "
                       "{}: No heartbeat in {}s, "
                       "restarting Ray to recover...".format(node_id, delta))
        updater = NodeUpdaterThread(
            node_id=node_id,
            provider_config=self.config["provider"],
            provider=self.provider,
            auth_config=self.config["auth"],
            cluster_name=self.config["cluster_name"],
            file_mounts={},
            initialization_commands=[],
            setup_commands=with_head_node_ip(
                self.config["worker_start_ray_commands"]),
            runtime_hash=self.runtime_hash,
            process_runner=self.process_runner,
            use_internal_ip=True)
        updater.start()
        self.updaters[node_id] = updater

    def should_update(self, node_id):
        if not self.can_update(node_id):
            return (None, None)

        if self.files_up_to_date(node_id):
            return (None, None)

        successful_updated = self.num_successful_updates.get(node_id, 0) > 0
        if successful_updated and self.config.get("restart_only", False):
            init_commands = self.config["worker_start_ray_commands"]
        elif successful_updated and self.config.get("no_restart", False):
            init_commands = self.config["worker_setup_commands"]
        else:
            init_commands = (self.config["worker_setup_commands"] +
                             self.config["worker_start_ray_commands"])

        return (node_id, init_commands)

    def spawn_updater(self, node_id, init_commands):
        updater = NodeUpdaterThread(
            node_id=node_id,
            provider_config=self.config["provider"],
            provider=self.provider,
            auth_config=self.config["auth"],
            cluster_name=self.config["cluster_name"],
            file_mounts=self.config["file_mounts"],
            initialization_commands=with_head_node_ip(
                self.config["initialization_commands"]),
            setup_commands=with_head_node_ip(init_commands),
            runtime_hash=self.runtime_hash,
            process_runner=self.process_runner,
            use_internal_ip=True)
        updater.start()
        self.updaters[node_id] = updater

    def can_update(self, node_id):
        if node_id in self.updaters:
            return False
        if not self.launch_config_ok(node_id):
            return False
        if self.num_failed_updates.get(node_id, 0) > 0:  # TODO(ekl) retry?
            return False
        return True

    def launch_new_node(self, count, worker_name):
        logger.info(
            "StandardAutoscaler: Queue {} new nodes for launch".format(count))
        self.num_launches_pending.inc(count)
        config = copy.deepcopy(self.config)
        # TODO this is hacky
        config['worker_nodes'] = config['worker_nodes'][worker_name]
        config['worker_nodes']['SubnetIds'] = config['worker_nodes']['SubnetIds']
        config['worker_nodes']['SecurityGroupIds'] = config['worker_nodes']['SecurityGroupIds']
        self.launch_queue.put((config, count))

    def workers(self):
        return self.provider.non_terminated_nodes(
            tag_filters={TAG_RAY_NODE_TYPE: "worker"})

    def log_info_string(self, nodes, target):
        logger.info("StandardAutoscaler: {}".format(
            self.info_string(nodes, target)))
        logger.info("LoadMetrics: {}".format(self.load_metrics.info_string()))

    def info_string(self, nodes, target):
        suffix = ""
        if self.num_launches_pending:
            suffix += " ({} pending)".format(self.num_launches_pending.value)
        if self.updaters:
            suffix += " ({} updating)".format(len(self.updaters))
        if self.num_failed_updates:
            suffix += " ({} failed to update)".format(
                len(self.num_failed_updates))
        if self.bringup:
            suffix += " (bringup=True)"

        return "{}/{} target nodes{}".format(len(nodes), target, suffix)

    def publish_autoscaler_status(self, nodes, target):
        autoscaler_status = {"StandardAutoscaler": self.info_string(nodes, target)}
        self.redis_client.publish(self.redis_key, json.dumps(autoscaler_status))

    def request_resources(self, resources):
        for resource, count in resources.items():
            if isinstance(count, int):
                self.resource_requests[resource] = max(
                    self.resource_requests[resource], count)
            else:
                self.resource_requests[resource] = count

        logger.info("StandardAutoscaler: resource_requests={}".format(
            self.resource_requests))

    def kill_workers(self):
        logger.error("StandardAutoscaler: kill_workers triggered")
        nodes = self.workers()
        if nodes:
            self.provider.terminate_nodes(nodes)
        logger.error("StandardAutoscaler: terminated {} node(s)".format(
            len(nodes)))


def typename(v):
    if isinstance(v, type):
        return v.__name__
    else:
        return type(v).__name__


def check_required(config, schema):
    # Check required schema entries
    if not isinstance(config, dict):
        raise ValueError("Config is not a dictionary")

    for k, (v, kreq) in schema.items():
        if v is None:
            continue  # None means we don't validate the field
        if kreq is REQUIRED:
            if k not in config:
                type_str = typename(v)
                raise ValueError(
                    "Missing required config key `{}` of type {}".format(
                        k, type_str))
            if not isinstance(v, type):
                check_required(config[k], v)


def check_extraneous(config, schema):
    """Make sure all items of config are in schema"""
    if not isinstance(config, dict):
        raise ValueError("Config {} is not a dictionary".format(config))
    for k in config:
        if k not in schema:
            raise ValueError("Unexpected config key `{}` not in {}".format(
                k, list(schema.keys())))
        v, kreq = schema[k]
        if v is None:
            continue
        elif isinstance(v, type):
            if not isinstance(config[k], v):
                if v is str and isinstance(config[k], string_types):
                    continue
                raise ValueError(
                    "Config key `{}` has wrong type {}, expected {}".format(
                        k,
                        type(config[k]).__name__, v.__name__))
        else:
            check_extraneous(config[k], v)


def validate_config(config, schema=CLUSTER_CONFIG_SCHEMA):
    """Required Dicts indicate that no extra fields can be introduced."""
    if not isinstance(config, dict):
        raise ValueError("Config {} is not a dictionary".format(config))

    check_required(config, schema)
    check_extraneous(config, schema)


def fillout_defaults(config):
    defaults = get_default_config(config["provider"])
    defaults.update(config)
    merge_setup_commands(defaults)
    dockerize_if_needed(defaults)
    return defaults


def merge_setup_commands(config):
    config["head_setup_commands"] = (
        config["setup_commands"] + config["head_setup_commands"])
    config["worker_setup_commands"] = (
        config["setup_commands"] + config["worker_setup_commands"])
    return config


def with_head_node_ip(cmds):
    head_ip = services.get_node_ip_address()
    out = []
    for cmd in cmds:
        out.append("export RAY_HEAD_IP={}; {}".format(head_ip, cmd))
    return out


def hash_launch_conf(node_conf, auth):
    hasher = hashlib.sha1()
    hasher.update(
        json.dumps([node_conf, auth], sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


# Cache the file hashes to avoid rescanning it each time. Also, this avoids
# inadvertently restarting workers if the file mount content is mutated on the
# head node.
_hash_cache = {}


def hash_runtime_conf(file_mounts, extra_objs):
    hasher = hashlib.sha1()

    def add_content_hashes(path):
        def add_hash_of_file(fpath):
            with open(fpath, "rb") as f:
                for chunk in iter(lambda: f.read(2**20), b""):
                    hasher.update(chunk)

        path = os.path.expanduser(path)
        if os.path.isdir(path):
            dirs = []
            for dirpath, _, filenames in os.walk(path):
                dirs.append((dirpath, sorted(filenames)))
            for dirpath, filenames in sorted(dirs):
                hasher.update(dirpath.encode("utf-8"))
                for name in filenames:
                    hasher.update(name.encode("utf-8"))
                    fpath = os.path.join(dirpath, name)
                    add_hash_of_file(fpath)
        else:
            add_hash_of_file(path)

    conf_str = (json.dumps(file_mounts, sort_keys=True).encode("utf-8") +
                json.dumps(extra_objs, sort_keys=True).encode("utf-8"))

    # Important: only hash the files once. Otherwise, we can end up restarting
    # workers if the files were changed and we re-hashed them.
    if conf_str not in _hash_cache:
        hasher.update(conf_str)
        for local_path in sorted(file_mounts.values()):
            add_content_hashes(local_path)
        _hash_cache[conf_str] = hasher.hexdigest()

    return _hash_cache[conf_str]


#def request_resources(num_cpus=None, num_gpus=None, worker_type=None):
#    """Remotely request some CPU or GPU resources from the autoscaler.
#
#    This function is to be called e.g. on a node before submitting a bunch of
#    ray.remote calls to ensure that resources rapidly become available.
#
#    In the future this could be extended to do GPU cores or other custom
#    resources.
#
#    This function is non blocking.
#
#    Args:
#
#        num_cpus: int -- the number of CPU cores to request
#        num_gpus: int -- the number of GPUs to request (Not implemented)
#
#    """
#    if num_gpus is not None:
#        raise NotImplementedError(
#            "GPU resource is not yet supported through request_resources")
#    r = services.create_redis_client(
#        global_worker.node.redis_address,
#        password=global_worker.node.redis_password)
#    assert isinstance(num_cpus, int)
#    if num_cpus > 0:
#        r.publish(AUTOSCALER_RESOURCE_REQUEST_CHANNEL,
#                  json.dumps({
#                      "CPU": num_cpus
#                  }))
#    if worker_type is not None:
#        r.publish(AUTOSCALER_RESOURCE_REQUEST_CHANNEL,
#                  json.dumps({
#                      "worker_type": worker_type
#                  }))

#def _request_resources(num_cpus=None, num_gpus=None):
#    """Remotely request some CPU or GPU resources from the autoscaler.
#    This function is to be called e.g. on a node before submitting a bunch of
#    ray.remote calls to ensure that resources rapidly become available.
#    In the future this could be extended to do GPU cores or other custom
#    resources.
#    This function is non blocking.
#    Args:
#        num_cpus: int -- the number of CPU cores to request
#        num_gpus: int -- the number of GPUs to request (Not implemented)
#    """
#    if num_gpus is not None:
#        raise NotImplementedError(
#            "GPU resource is not yet supported through request_resources")
#    r = services.create_redis_client(
#        global_worker.node.redis_address,
#        password=global_worker.node.redis_password)
#    assert isinstance(num_cpus, int)
#    if num_cpus > 0:
#        r.publish(AUTOSCALER_RESOURCE_REQUEST_CHANNEL,
#                  json.dumps({
#                      "CPU": num_cpus
#                  }))

def request_resources(resources):

    r = services.create_redis_client(
        global_worker.node.redis_address,
        password=global_worker.node.redis_password)

    r.publish(AUTOSCALER_RESOURCE_REQUEST_CHANNEL,
              json.dumps({resource_type: resource_count
                          for resource_type, resource_count in resources.items()})

