# src/ai/object_pool.py
import threading
from typing import List, Optional
from ..core.position import Position

class PositionPool:
    """Thread-safe Position object pool for memory optimization"""
    
    def __init__(self, initial_size: int = 100, max_size: int = 1000):
        self._available: List[Position] = []
        self._in_use = set()
        self._lock = threading.Lock()
        self._max_size = max_size
        
        # Pre-allocate initial positions
        for _ in range(initial_size):
            self._available.append(Position())
    
    def get(self) -> Position:
        """Get a Position object from the pool"""
        with self._lock:
            if self._available:
                pos = self._available.pop()
                self._in_use.add(id(pos))
                return pos
            else:
                # Create new one if pool is empty
                pos = Position()
                self._in_use.add(id(pos))
                return pos
    
    def release(self, pos: Position):
        """Return a Position object to the pool"""
        if pos is None:
            return
            
        pos_id = id(pos)
        with self._lock:
            if pos_id in self._in_use:
                self._in_use.remove(pos_id)
                
                # Only add back to pool if under max size
                if len(self._available) < self._max_size:
                    # Reset the position to initial state
                    self._reset_position(pos)
                    self._available.append(pos)
    
    def _reset_position(self, pos: Position):
        """Reset position to clean state for reuse"""
        # Reset all bitboards to 0
        for color in range(2):
            for piece_type in range(6):
                pos.pieces[color][piece_type] = 0
        
        pos.side = 0  # White to move
        pos.enpas = 0
        pos.castle = 0
        pos.hash_key = 0
    
    def get_stats(self) -> dict:
        """Get pool statistics"""
        with self._lock:
            return {
                'available': len(self._available),
                'in_use': len(self._in_use),
                'total': len(self._available) + len(self._in_use)
            }

# Global pool instance
_position_pool = PositionPool()

def get_position() -> Position:
    """Get a position from the global pool"""
    return _position_pool.get()

def release_position(pos: Position):
    """Release a position back to the global pool"""
    _position_pool.release(pos)

def get_pool_stats() -> dict:
    """Get global pool statistics"""
    return _position_pool.get_stats()
