"""
Service Manager for BrahminyKite Tools.

Manages lifecycle of all gRPC services with health checks and monitoring.
"""

import os
import yaml
import time
import signal
import asyncio
import threading
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from .services import (
    serve_empirical,
    # serve_contextual,
    # serve_consistency,
    # serve_power,
    # serve_utility,
    # serve_evolution
)


class ServiceManager:
    """Manages all gRPC tool services."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.services = {}
        self.health_checkers = {}
        self.running = False
        self._shutdown_event = threading.Event()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load service configuration."""
        if not config_path:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                'configs/service_config.yaml'
            )
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, shutting down...")
        self.stop_all()
    
    def start_service(self, name: str, serve_func, port: int, config: Dict[str, Any]):
        """Start a single service."""
        try:
            print(f"Starting {name} service on port {port}...")
            server = serve_func(port=port, config=config)
            self.services[name] = server
            
            # Setup health checker
            self._setup_health_check(name, port)
            
            print(f"✓ {name} service started successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to start {name} service: {e}")
            return False
    
    def _setup_health_check(self, name: str, port: int):
        """Setup health checking for a service."""
        channel = grpc.insecure_channel(f'localhost:{port}')
        health_stub = health_pb2_grpc.HealthStub(channel)
        self.health_checkers[name] = health_stub
    
    def check_health(self, name: str) -> bool:
        """Check if a service is healthy."""
        if name not in self.health_checkers:
            return False
        
        try:
            request = health_pb2.HealthCheckRequest(service=name)
            response = self.health_checkers[name].Check(request, timeout=1)
            return response.status == health_pb2.HealthCheckResponse.SERVING
        except:
            return False
    
    def start_all(self):
        """Start all configured services."""
        self.running = True
        
        service_configs = self.config['services']
        
        # Start empirical service
        if 'empirical' in service_configs:
            cfg = service_configs['empirical']
            self.start_service('empirical', serve_empirical, cfg['port'], cfg)
        
        # Start contextual service
        # if 'contextual' in service_configs:
        #     cfg = service_configs['contextual']
        #     self.start_service('contextual', serve_contextual, cfg['port'], cfg)
        
        # Start consistency service
        # if 'consistency' in service_configs:
        #     cfg = service_configs['consistency']
        #     self.start_service('consistency', serve_consistency, cfg['port'], cfg)
        
        # Start power dynamics service
        # if 'power_dynamics' in service_configs:
        #     cfg = service_configs['power_dynamics']
        #     self.start_service('power_dynamics', serve_power, cfg['port'], cfg)
        
        # Start utility service
        # if 'utility' in service_configs:
        #     cfg = service_configs['utility']
        #     self.start_service('utility', serve_utility, cfg['port'], cfg)
        
        # Start evolution service
        # if 'evolution' in service_configs:
        #     cfg = service_configs['evolution']
        #     self.start_service('evolution', serve_evolution, cfg['port'], cfg)
        
        print(f"\n✓ Started {len(self.services)} services")
        
        # Start monitoring thread
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start service monitoring."""
        def monitor():
            while self.running:
                time.sleep(10)  # Check every 10 seconds
                
                for name in self.services:
                    if not self.check_health(name):
                        print(f"⚠ Service {name} is unhealthy")
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def stop_all(self):
        """Stop all services gracefully."""
        self.running = False
        
        for name, server in self.services.items():
            print(f"Stopping {name} service...")
            server.stop(grace=5)
        
        self.services.clear()
        self.health_checkers.clear()
        self._shutdown_event.set()
    
    def wait_for_termination(self):
        """Wait for all services to terminate."""
        try:
            self._shutdown_event.wait()
        except KeyboardInterrupt:
            self.stop_all()
    
    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services."""
        status = {}
        
        for name, server in self.services.items():
            status[name] = {
                'running': True,
                'healthy': self.check_health(name),
                'port': self.config['services'][name]['port']
            }
        
        return status


class ServiceClient:
    """Unified client for all services with load balancing."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.clients = {}
        self._init_clients()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load client configuration."""
        if not config_path:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                'configs/service_config.yaml'
            )
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_clients(self):
        """Initialize clients for all services."""
        from .clients import (
            EmpiricalToolsClient,
            # ContextualToolsClient,
            # ConsistencyToolsClient,
            # PowerDynamicsToolsClient,
            # UtilityToolsClient,
            # EvolutionToolsClient
        )
        
        services = self.config['services']
        
        if 'empirical' in services:
            self.clients['empirical'] = EmpiricalToolsClient(
                port=services['empirical']['port']
            )
        
        # if 'contextual' in services:
        #     self.clients['contextual'] = ContextualToolsClient(
        #         port=services['contextual']['port']
        #     )
        
        # Add other clients...
    
    def get_client(self, framework: str):
        """Get client for a specific framework."""
        return self.clients.get(framework)
    
    async def verify_claim(self, claim: str, frameworks: List[str] = None):
        """Verify a claim using multiple frameworks in parallel."""
        if not frameworks:
            frameworks = list(self.clients.keys())
        
        tasks = []
        for framework in frameworks:
            client = self.clients.get(framework)
            if client:
                # Add framework-specific verification tasks
                if framework == 'empirical':
                    tasks.append(
                        asyncio.create_task(
                            asyncio.to_thread(
                                client.check_logical_consistency,
                                claim
                            )
                        )
                    )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            framework: result 
            for framework, result in zip(frameworks, results)
            if not isinstance(result, Exception)
        }
    
    def close_all(self):
        """Close all client connections."""
        for client in self.clients.values():
            if hasattr(client, 'close'):
                client.close()


if __name__ == '__main__':
    # Start all services
    manager = ServiceManager()
    manager.start_all()
    
    print("\nServices running. Press Ctrl+C to stop.")
    manager.wait_for_termination()