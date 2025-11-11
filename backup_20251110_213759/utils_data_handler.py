"""
Enhanced Data Handler with Redis Caching and Async Processing
Phase 1 Implementation: High-Performance Data Pipeline
"""
from __future__ import annotations

import asyncio
import os
import pickle
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np
import redis
from concurrent.futures import ThreadPoolExecutor
import logging

# Import configuration
from config import (
    REDIS_CONFIG, DATA_PIPELINE_CONFIG, PERFORMANCE_CONFIG,
    DATA_DIR, LIVE_DATA_FILE, TRADES_FILE, SIGNALS_FILE,
    OPTIMIZATION_FLAGS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Data class for performance metrics"""
    data_processing_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    total_bars_processed: int = 0
    total_cache_hits: int = 0
    total_cache_misses: int = 0

class EnhancedDataHandler:
    """
    High-performance data handler with Redis caching and async processing
    Phase 1: Optimized data pipeline for low-latency scalping operations
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern - ensures only one instance exists"""
        if cls._instance is None:
            cls._instance = super(EnhancedDataHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: str = "data/nq_live.db"):
        if self._initialized:
            return
            
        self.db_path = db_path
        self.performance_metrics = PerformanceMetrics()
        
        # Initialize Redis connection if enabled
        self.redis_client = None
        if OPTIMIZATION_FLAGS.get('use_redis_cache', False):
            try:
                self.redis_client = redis.Redis(**REDIS_CONFIG)
                self.redis_client.ping()  # Test connection
                logger.info("✅ Redis connection established")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}")
                self.redis_client = None
        
        # Initialize async components
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.batch_queue = []
        self.batch_timer = None
        
        # Performance tracking
        self._lock = threading.Lock()
        self._last_performance_update = datetime.now()
        
        # Initialize files
        self._init_files()
        
        self._initialized = True
        logger.info("✅ Enhanced Data Handler initialized with optimization features")
    
    def _init_files(self):
        """Initialize CSV files with proper headers"""
        files_config = {
            TRADES_FILE: [
                "timestamp", "ticker", "action", "price", "size", "signal", 
                "stop_loss", "take_profit", "pnl", "status", "entry_price", 
                "exit_price", "entry_time", "exit_time", "r_multiple"
            ],
            SIGNALS_FILE: [
                "timestamp", "strategy", "signal", "confidence", "price", 
                "target", "stop", "position_size", "market_context"
            ],
            PERFORMANCE_FILE: [
                "timestamp", "total_pnl", "win_rate", "profit_factor", 
                "max_drawdown", "total_trades", "latency_ms", "cache_hit_rate"
            ]
        }
        
        for file_path, headers in files_config.items():
            if not file_path.exists():
                pd.DataFrame(columns=headers).to_csv(file_path, index=False)
                logger.info(f"Created {file_path.name}")
    
    # =========================================================================
    # REDIS CACHE OPERATIONS
    # =========================================================================
    
    def _get_cache_key(self, prefix: str, *args) -> str:
        """Generate Redis cache key"""
        return f"nq_trading:{prefix}:{':'.join(str(arg) for arg in args)}"
    
    def _cache_data(self, key: str, data: Any, ttl: int = None) -> bool:
        """Cache data in Redis with optional TTL"""
        if not self.redis_client:
            return False
            
        try:
            serialized_data = pickle.dumps(data)
            if ttl:
                return self.redis_client.setex(key, ttl, serialized_data)
            else:
                return self.redis_client.set(key, serialized_data)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data from Redis"""
        if not self.redis_client:
            return None
            
        try:
            data = self.redis_client.get(key)
            if data:
                self.performance_metrics.total_cache_hits += 1
                return pickle.loads(data)
            else:
                self.performance_metrics.total_cache_misses += 1
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def _update_cache_stats(self):
        """Update cache hit rate statistics"""
        total_requests = self.performance_metrics.total_cache_hits + self.performance_metrics.total_cache_misses
        if total_requests > 0:
            self.performance_metrics.cache_hit_rate = self.performance_metrics.total_cache_hits / total_requests
    
    # =========================================================================
    # ASYNCHRONOUS BATCH PROCESSING
    # =========================================================================
    
    async def _process_batch_async(self, batch_data: List[Dict]):
        """Process batch data asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._write_batch_sync, batch_data)
    
    def _write_batch_sync(self, batch_data: List[Dict]):
        """Synchronous batch write operation"""
        if not batch_data:
            return
            
        try:
            df = pd.DataFrame(batch_data)
            file_path = batch_data[0].get('_file_path', LIVE_DATA_FILE)
            
            # Remove internal metadata
            df = df.drop(columns=['_file_path'], errors='ignore')
            
            # Append to CSV file
            df.to_csv(file_path, mode='a', header=False, index=False)
            
            logger.debug(f"Batch written {len(batch_data)} records to {file_path.name}")
            
        except Exception as e:
            logger.error(f"Batch write error: {e}")
    
    def _schedule_batch_processing(self):
        """Schedule batch processing with timer"""
        if self.batch_timer:
            self.batch_timer.cancel()
        
        if self.batch_queue:
            self.batch_timer = threading.Timer(
                DATA_PIPELINE_CONFIG['batch_write_interval'],
                self._flush_batch
            )
            self.batch_timer.start()
    
    def _flush_batch(self):
        """Flush batch queue to storage"""
        with self._lock:
            if self.batch_queue:
                batch_data = self.batch_queue.copy()
                self.batch_queue.clear()
                
                # Process asynchronously
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self._process_batch_async(batch_data))
                    loop.close()
                except Exception as e:
                    logger.error(f"Async batch processing error: {e}")
    
    # =========================================================================
    # CORE DATA OPERATIONS
    # =========================================================================
    
    def add_bar(self, bar_data: Dict) -> bool:
        """
        Add a new bar with optimized processing
        Returns processing latency in milliseconds
        """
        start_time = time.time()
        
        try:
            # Validate and prepare data
            ts = pd.to_datetime(bar_data.get("timestamp"), errors="coerce", utc=True)
            if pd.isna(ts):
                logger.warning(f"Invalid timestamp: {bar_data.get('timestamp')}")
                return False
            
            # Prepare record
            record = {
                "timestamp": ts.isoformat(),
                "open": float(bar_data.get("open", 0)),
                "high": float(bar_data.get("high", 0)),
                "low": float(bar_data.get("low", 0)),
                "close": float(bar_data.get("close", 0)),
                "volume": float(bar_data.get("volume", 0)),
                "_file_path": str(LIVE_DATA_FILE)
            }
            
            # Add to batch queue for async processing
            with self._lock:
                self.batch_queue.append(record)
                self.performance_metrics.total_bars_processed += 1
            
            # Cache the latest bar
            cache_key = self._get_cache_key("latest_bar")
            self._cache_data(cache_key, record, DATA_PIPELINE_CONFIG['cache_ttl_seconds'])
            
            # Schedule batch processing
            self._schedule_batch_processing()
            
            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.performance_metrics.data_processing_latency_ms = processing_time
            
            logger.debug(f"Bar processed in {processing_time:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"Error adding bar: {e}")
            return False
    
    def get_latest_bars(self, n: int = 150) -> pd.DataFrame:
        """Get latest bars with caching optimization"""
        cache_key = self._get_cache_key("bars", n)
        
        # Try to get from cache first
        if OPTIMIZATION_FLAGS.get('use_redis_cache', False):
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                return pd.DataFrame(cached_data)
        
        # Fallback to file-based retrieval
        try:
            if LIVE_DATA_FILE.exists():
                df = pd.read_csv(LIVE_DATA_FILE)
                if not df.empty:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
                    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
                    result_df = df.tail(n) if len(df) > n else df
                    
                    # Cache the result
                    if OPTIMIZATION_FLAGS.get('use_redis_cache', False):
                        self._cache_data(cache_key, result_df.to_dict('records'), 60)
                    
                    return result_df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting latest bars: {e}")
            return pd.DataFrame()
    
    def append_signal(self, signal_data: Dict) -> bool:
        """Append trading signal with batch processing"""
        try:
            record = {
                "timestamp": signal_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "strategy": signal_data.get("strategy", "unknown"),
                "signal": signal_data.get("signal", ""),
                "confidence": float(signal_data.get("confidence", 0.0)),
                "price": float(signal_data.get("price", 0.0)),
                "target": signal_data.get("target"),
                "stop": signal_data.get("stop"),
                "position_size": signal_data.get("position_size", 1),
                "market_context": str(signal_data.get("market_context", {})),
                "_file_path": str(SIGNALS_FILE)
            }
            
            with self._lock:
                self.batch_queue.append(record)
            
            self._schedule_batch_processing()
            return True
            
        except Exception as e:
            logger.error(f"Error appending signal: {e}")
            return False
    
    def append_trade(self, trade_data: Dict) -> bool:
        """Append trade record with batch processing"""
        try:
            record = {
                "timestamp": trade_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "ticker": trade_data.get("ticker", "NQ"),
                "action": trade_data.get("action", ""),
                "price": float(trade_data.get("price", 0.0)),
                "size": int(trade_data.get("size", 1)),
                "signal": trade_data.get("signal", ""),
                "stop_loss": trade_data.get("stop_loss"),
                "take_profit": trade_data.get("take_profit"),
                "pnl": float(trade_data.get("pnl", 0.0)),
                "status": trade_data.get("status", ""),
                "entry_price": trade_data.get("entry_price"),
                "exit_price": trade_data.get("exit_price"),
                "entry_time": trade_data.get("entry_time"),
                "exit_time": trade_data.get("exit_time"),
                "r_multiple": trade_data.get("r_multiple"),
                "_file_path": str(TRADES_FILE)
            }
            
            with self._lock:
                self.batch_queue.append(record)
            
            self._schedule_batch_processing()
            return True
            
        except Exception as e:
            logger.error(f"Error appending trade: {e}")
            return False
    
    # =========================================================================
    # PERFORMANCE MONITORING
    # =========================================================================
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        self._update_cache_stats()
        
        return {
            'data_processing_latency_ms': self.performance_metrics.data_processing_latency_ms,
            'cache_hit_rate': self.performance_metrics.cache_hit_rate,
            'memory_usage_mb': self.performance_metrics.memory_usage_mb,
            'total_bars_processed': self.performance_metrics.total_bars_processed,
            'total_cache_hits': self.performance_metrics.total_cache_hits,
            'total_cache_misses': self.performance_metrics.total_cache_misses,
            'batch_queue_size': len(self.batch_queue)
        }
    
    def update_performance_metrics(self):
        """Update and log performance metrics"""
        current_time = datetime.now()
        if (current_time - self._last_performance_update).seconds >= PERFORMANCE_CONFIG['metrics_update_interval']:
            metrics = self.get_performance_metrics()
            
            # Log performance data
            performance_record = {
                'timestamp': current_time.isoformat(),
                'data_processing_latency_ms': metrics['data_processing_latency_ms'],
                'cache_hit_rate': metrics['cache_hit_rate'],
                'memory_usage_mb': metrics['memory_usage_mb'],
                'total_bars_processed': metrics['total_bars_processed'],
                'batch_queue_size': metrics['batch_queue_size']
            }
            
            # Append to performance file
            self.batch_queue.append({
                **performance_record,
                '_file_path': str(PERFORMANCE_FILE)
            })
            
            self._last_performance_update = current_time
            logger.info(f"Performance metrics updated: {metrics}")
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def flush_all_data(self):
        """Flush all pending data to storage"""
        self._flush_batch()
        logger.info("All data flushed to storage")
    
    def clear_cache(self):
        """Clear Redis cache"""
        if self.redis_client:
            self.redis_client.flushdb()
            logger.info("Redis cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        if not self.redis_client:
            return {'status': 'disabled'}
        
        try:
            info = self.redis_client.info()
            return {
                'status': 'active',
                'used_memory_mb': info.get('used_memory_rss', 0) / (1024 * 1024),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'status': 'error', 'error': str(e)}


# Global singleton instance
enhanced_data_handler = EnhancedDataHandler()

# Convenience functions for backward compatibility
def add_bar(bar_data: Dict) -> bool:
    """Add bar data (backward compatibility)"""
    return enhanced_data_handler.add_bar(bar_data)

def get_latest_bars(n: int = 150) -> pd.DataFrame:
    """Get latest bars (backward compatibility)"""
    return enhanced_data_handler.get_latest_bars(n)

def append_signal(signal_data: Dict) -> bool:
    """Append signal (backward compatibility)"""
    return enhanced_data_handler.append_signal(signal_data)

def append_trade(trade_data: Dict) -> bool:
    """Append trade (backward compatibility)"""
    return enhanced_data_handler.append_trade(trade_data)