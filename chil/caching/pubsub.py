"""
Redis pub/sub implementation for cache synchronization and messaging.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Callable, Optional, List, Union
from datetime import datetime
import traceback

from .client import RedisClient, get_redis_client

logger = logging.getLogger(__name__)

# Type alias for message handlers
MessageHandler = Callable[[Dict[str, Any]], None]


class RedisPubSub:
    """
    Redis pub/sub manager for distributed cache synchronization.
    
    Features:
    - Message routing by channel
    - JSON serialization/deserialization
    - Error handling and retries
    - Message filtering
    - Statistics tracking
    """
    
    def __init__(self, client: Optional[RedisClient] = None):
        self.client = client
        self._pubsub = None
        self._subscribers: Dict[str, List[MessageHandler]] = {}
        self._running = False
        self._listener_task: Optional[asyncio.Task] = None
        
        # Message filtering
        self._filters: Dict[str, List[Callable[[Dict[str, Any]], bool]]] = {}
        
        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_filtered": 0,
            "errors": 0,
            "channels": 0
        }
    
    async def initialize(self):
        """Initialize the pub/sub system."""
        if self.client is None:
            self.client = await get_redis_client()
        
        self._pubsub = self.client.client.pubsub()
        logger.info("Redis pub/sub initialized")
    
    async def start(self):
        """Start the pub/sub listener."""
        if self._running:
            return
        
        await self.initialize()
        self._running = True
        
        # Start listener task
        self._listener_task = asyncio.create_task(self._listen_loop())
        logger.info("Redis pub/sub listener started")
    
    async def stop(self):
        """Stop the pub/sub listener."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel listener task
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        
        # Close pubsub connection
        if self._pubsub:
            await self._pubsub.close()
        
        logger.info("Redis pub/sub listener stopped")
    
    async def subscribe(self, channel: str, handler: MessageHandler):
        """
        Subscribe to a channel with a message handler.
        
        Args:
            channel: Channel name to subscribe to
            handler: Function to handle received messages
        """
        if channel not in self._subscribers:
            self._subscribers[channel] = []
            
            # Subscribe to the channel
            if self._pubsub:
                await self._pubsub.subscribe(channel)
                self._stats["channels"] += 1
                logger.info(f"Subscribed to channel: {channel}")
        
        self._subscribers[channel].append(handler)
    
    async def unsubscribe(self, channel: str, handler: Optional[MessageHandler] = None):
        """
        Unsubscribe from a channel.
        
        Args:
            channel: Channel name to unsubscribe from
            handler: Specific handler to remove (if None, removes all)
        """
        if channel not in self._subscribers:
            return
        
        if handler:
            # Remove specific handler
            if handler in self._subscribers[channel]:
                self._subscribers[channel].remove(handler)
        else:
            # Remove all handlers
            self._subscribers[channel].clear()
        
        # If no handlers left, unsubscribe from channel
        if not self._subscribers[channel]:
            del self._subscribers[channel]
            
            if self._pubsub:
                await self._pubsub.unsubscribe(channel)
                self._stats["channels"] -= 1
                logger.info(f"Unsubscribed from channel: {channel}")
    
    async def publish(self, channel: str, message: Union[Dict[str, Any], str, bytes]):
        """
        Publish a message to a channel.
        
        Args:
            channel: Channel name to publish to
            message: Message to publish (will be JSON serialized if dict)
        """
        if not self.client:
            await self.initialize()
        
        try:
            # Serialize message if needed
            if isinstance(message, dict):
                message_data = json.dumps({
                    **message,
                    "_timestamp": datetime.now().isoformat(),
                    "_channel": channel
                })
            elif isinstance(message, str):
                message_data = message
            else:
                message_data = message
            
            # Publish message
            await self.client.client.publish(channel, message_data)
            self._stats["messages_sent"] += 1
            
            logger.debug(f"Published message to {channel}")
        
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Failed to publish message to {channel}: {e}")
            raise
    
    async def add_filter(self, channel: str, filter_func: Callable[[Dict[str, Any]], bool]):
        """
        Add a message filter for a channel.
        
        Args:
            channel: Channel name
            filter_func: Function that returns True if message should be processed
        """
        if channel not in self._filters:
            self._filters[channel] = []
        
        self._filters[channel].append(filter_func)
        logger.debug(f"Added filter for channel {channel}")
    
    async def _listen_loop(self):
        """Main listening loop for pub/sub messages."""
        try:
            while self._running:
                try:
                    # Get next message
                    message = await self._pubsub.get_message(timeout=1.0)
                    
                    if message is None:
                        continue
                    
                    # Skip subscription messages
                    if message['type'] not in ['message', 'pmessage']:
                        continue
                    
                    await self._handle_message(message)
                
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self._stats["errors"] += 1
                    logger.error(f"Error in pub/sub listener: {e}")
                    await asyncio.sleep(1)  # Brief pause before retrying
        
        except asyncio.CancelledError:
            logger.info("Pub/sub listener cancelled")
        except Exception as e:
            logger.error(f"Fatal error in pub/sub listener: {e}")
    
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle a received pub/sub message."""
        channel = message['channel'].decode() if isinstance(message['channel'], bytes) else message['channel']
        data = message['data']
        
        # Skip if no subscribers
        if channel not in self._subscribers:
            return
        
        try:
            # Parse message data
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            if isinstance(data, str):
                try:
                    parsed_data = json.loads(data)
                except json.JSONDecodeError:
                    parsed_data = {"raw_data": data}
            else:
                parsed_data = data
            
            # Apply filters
            if channel in self._filters:
                for filter_func in self._filters[channel]:
                    if not filter_func(parsed_data):
                        self._stats["messages_filtered"] += 1
                        return
            
            # Call handlers
            for handler in self._subscribers[channel]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(parsed_data)
                    else:
                        handler(parsed_data)
                except Exception as e:
                    logger.error(f"Error in message handler for {channel}: {e}")
                    logger.debug(traceback.format_exc())
            
            self._stats["messages_received"] += 1
            logger.debug(f"Processed message from {channel}")
        
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Error handling message from {channel}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pub/sub statistics."""
        return {
            **self._stats,
            "active_channels": list(self._subscribers.keys()),
            "running": self._running
        }
    
    def reset_stats(self):
        """Reset pub/sub statistics."""
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_filtered": 0,
            "errors": 0,
            "channels": len(self._subscribers)
        }


class MessageBroker:
    """
    High-level message broker with routing and patterns.
    """
    
    def __init__(self, pubsub: Optional[RedisPubSub] = None):
        self.pubsub = pubsub
        self._routes: Dict[str, List[str]] = {}  # topic -> channels
        self._middleware: List[Callable] = []
    
    async def initialize(self):
        """Initialize the message broker."""
        if self.pubsub is None:
            self.pubsub = RedisPubSub()
        
        await self.pubsub.initialize()
        await self.pubsub.start()
    
    async def add_route(self, topic: str, channels: List[str]):
        """Add routing from topic to channels."""
        self._routes[topic] = channels
        logger.info(f"Added route: {topic} -> {channels}")
    
    async def subscribe_to_topic(self, topic: str, handler: MessageHandler):
        """Subscribe to a topic (routes to appropriate channels)."""
        channels = self._routes.get(topic, [topic])  # Default to topic as channel
        
        for channel in channels:
            await self.pubsub.subscribe(channel, handler)
    
    async def publish_to_topic(self, topic: str, message: Dict[str, Any]):
        """Publish message to a topic (routes to appropriate channels)."""
        channels = self._routes.get(topic, [topic])
        
        # Add middleware processing
        processed_message = message
        for middleware in self._middleware:
            processed_message = await middleware(processed_message)
        
        # Publish to all channels
        for channel in channels:
            await self.pubsub.publish(channel, processed_message)
    
    def add_middleware(self, middleware: Callable):
        """Add middleware for message processing."""
        self._middleware.append(middleware)


class CacheEventBroker(MessageBroker):
    """
    Specialized message broker for cache events.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._event_handlers: Dict[str, List[MessageHandler]] = {}
    
    async def initialize(self):
        """Initialize with cache-specific channels."""
        await super().initialize()
        
        # Set up cache event channels
        await self.add_route("cache.invalidation", ["cache_invalidation"])
        await self.add_route("cache.update", ["cache_update"])
        await self.add_route("cache.miss", ["cache_miss"])
        await self.add_route("cache.hit", ["cache_hit"])
    
    async def on_cache_invalidation(self, handler: MessageHandler):
        """Subscribe to cache invalidation events."""
        await self.subscribe_to_topic("cache.invalidation", handler)
    
    async def on_cache_update(self, handler: MessageHandler):
        """Subscribe to cache update events."""
        await self.subscribe_to_topic("cache.update", handler)
    
    async def on_cache_miss(self, handler: MessageHandler):
        """Subscribe to cache miss events."""
        await self.subscribe_to_topic("cache.miss", handler)
    
    async def on_cache_hit(self, handler: MessageHandler):
        """Subscribe to cache hit events."""
        await self.subscribe_to_topic("cache.hit", handler)
    
    async def emit_invalidation(self, keys: List[str], strategy: str = "immediate"):
        """Emit cache invalidation event."""
        await self.publish_to_topic("cache.invalidation", {
            "event": "invalidation",
            "keys": keys,
            "strategy": strategy,
            "timestamp": datetime.now().isoformat()
        })
    
    async def emit_cache_update(self, key: str, ttl: Optional[int] = None):
        """Emit cache update event."""
        await self.publish_to_topic("cache.update", {
            "event": "update",
            "key": key,
            "ttl": ttl,
            "timestamp": datetime.now().isoformat()
        })
    
    async def emit_cache_miss(self, key: str, reason: str = "not_found"):
        """Emit cache miss event."""
        await self.publish_to_topic("cache.miss", {
            "event": "miss",
            "key": key,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
    
    async def emit_cache_hit(self, key: str, ttl_remaining: Optional[int] = None):
        """Emit cache hit event."""
        await self.publish_to_topic("cache.hit", {
            "event": "hit",
            "key": key,
            "ttl_remaining": ttl_remaining,
            "timestamp": datetime.now().isoformat()
        })


# Global instances
_pubsub: Optional[RedisPubSub] = None
_cache_broker: Optional[CacheEventBroker] = None


async def get_pubsub() -> RedisPubSub:
    """Get the global pub/sub instance."""
    global _pubsub
    
    if _pubsub is None:
        _pubsub = RedisPubSub()
        await _pubsub.start()
    
    return _pubsub


async def get_cache_event_broker() -> CacheEventBroker:
    """Get the global cache event broker."""
    global _cache_broker
    
    if _cache_broker is None:
        _cache_broker = CacheEventBroker()
        await _cache_broker.initialize()
    
    return _cache_broker