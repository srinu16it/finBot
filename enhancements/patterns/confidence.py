"""
Pattern confidence tracking and scoring system.

This module tracks historical performance of patterns and provides confidence scores.
"""

from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging

# Try to import duckdb, fallback to in-memory storage if not available
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logging.warning("DuckDB not available, using in-memory pattern tracking")

logger = logging.getLogger(__name__)


class PatternConfidenceEngine:
    """
    Tracks pattern performance and provides confidence scores.
    
    Features:
    - Historical win rate tracking
    - Pattern-specific confidence scores
    - Performance metrics calculation
    - Statistical analysis
    """
    
    def __init__(self, db_path: str = "enhancements/cache/pattern_confidence.db"):
        """
        Initialize the confidence engine.
        
        Args:
            db_path: Path to the pattern confidence database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.use_duckdb = DUCKDB_AVAILABLE
        
        # In-memory storage when DuckDB is not available
        self.memory_storage = {
            "pattern_outcomes": [],
            "pattern_stats": {}
        }
        
        if self.use_duckdb:
            self._init_db()
        else:
            self._load_memory_storage()
    
    def _init_db(self):
        """Initialize database tables for pattern tracking."""
        with duckdb.connect(str(self.db_path)) as conn:
            # Pattern outcomes table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pattern_outcomes (
                    id INTEGER PRIMARY KEY,
                    pattern_name VARCHAR,
                    direction VARCHAR,
                    detected_date TIMESTAMP,
                    outcome_date TIMESTAMP,
                    success BOOLEAN,
                    price_at_detection FLOAT,
                    price_at_outcome FLOAT,
                    notes TEXT
                )
            """)
            
            # Pattern statistics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pattern_stats (
                    pattern_name VARCHAR,
                    direction VARCHAR,
                    total_occurrences INTEGER,
                    successful_outcomes INTEGER,
                    win_rate FLOAT,
                    avg_gain_loss FLOAT,
                    sharpe_ratio FLOAT,
                    last_updated TIMESTAMP,
                    PRIMARY KEY (pattern_name, direction)
                )
            """)
    
    def _load_memory_storage(self):
        """Load pattern data from JSON file for in-memory storage."""
        json_path = self.db_path.with_suffix('.json')
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    self.memory_storage = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load pattern data: {e}")
    
    def _save_memory_storage(self):
        """Save pattern data to JSON file."""
        if not self.use_duckdb:
            json_path = self.db_path.with_suffix('.json')
            try:
                with open(json_path, 'w') as f:
                    json.dump(self.memory_storage, f, indent=2, default=str)
            except Exception as e:
                logger.error(f"Failed to save pattern data: {e}")
    
    def record_pattern_detection(
        self,
        pattern_name: str,
        direction: str,
        price: float,
        detected_date: Optional[datetime] = None
    ) -> int:
        """
        Record a new pattern detection.
        
        Args:
            pattern_name: Name of the pattern (e.g., 'double_top')
            direction: Pattern direction ('bullish' or 'bearish')
            price: Price at detection
            detected_date: When pattern was detected (defaults to now)
            
        Returns:
            ID of the recorded detection
        """
        if detected_date is None:
            detected_date = datetime.now()
        
        if self.use_duckdb:
            with duckdb.connect(str(self.db_path)) as conn:
                result = conn.execute("""
                    INSERT INTO pattern_outcomes 
                    (pattern_name, direction, detected_date, price_at_detection)
                    VALUES (?, ?, ?, ?)
                    RETURNING id
                """, [pattern_name, direction, detected_date, price]).fetchone()
                return result[0]
        else:
            # In-memory storage
            detection = {
                "id": len(self.memory_storage["pattern_outcomes"]) + 1,
                "pattern_name": pattern_name,
                "direction": direction,
                "detected_date": detected_date.isoformat(),
                "price_at_detection": price,
                "outcome_date": None,
                "success": None,
                "price_at_outcome": None,
                "notes": None
            }
            self.memory_storage["pattern_outcomes"].append(detection)
            self._save_memory_storage()
            return detection["id"]
    
    def update_pattern_outcome(
        self,
        detection_id: int,
        success: bool,
        price_at_outcome: float,
        outcome_date: Optional[datetime] = None,
        notes: Optional[str] = None
    ):
        """
        Update the outcome of a pattern detection.
        
        Args:
            detection_id: ID of the pattern detection
            success: Whether the pattern played out successfully
            price_at_outcome: Price when outcome was determined
            outcome_date: When outcome was determined (defaults to now)
            notes: Optional notes about the outcome
        """
        if outcome_date is None:
            outcome_date = datetime.now()
        
        if self.use_duckdb:
            with duckdb.connect(str(self.db_path)) as conn:
                conn.execute("""
                    UPDATE pattern_outcomes
                    SET success = ?, price_at_outcome = ?, outcome_date = ?, notes = ?
                    WHERE id = ?
                """, [success, price_at_outcome, outcome_date, notes, detection_id])
        else:
            # Update in-memory storage
            for outcome in self.memory_storage["pattern_outcomes"]:
                if outcome["id"] == detection_id:
                    outcome["success"] = success
                    outcome["price_at_outcome"] = price_at_outcome
                    outcome["outcome_date"] = outcome_date.isoformat()
                    outcome["notes"] = notes
                    break
            self._save_memory_storage()
        
        # Update statistics
        self._update_pattern_statistics()
    
    def get_pattern_confidence(
        self,
        pattern_name: str,
        direction: str
    ) -> Dict[str, float]:
        """
        Get confidence metrics for a specific pattern.
        
        Args:
            pattern_name: Name of the pattern
            direction: Pattern direction ('bullish' or 'bearish')
            
        Returns:
            Dictionary with confidence metrics
        """
        if self.use_duckdb:
            with duckdb.connect(str(self.db_path)) as conn:
                result = conn.execute("""
                    SELECT win_rate, avg_gain_loss, sharpe_ratio, total_occurrences
                    FROM pattern_stats
                    WHERE pattern_name = ? AND direction = ?
                """, [pattern_name, direction]).fetchone()
                
                if result:
                    win_rate, avg_gain_loss, sharpe_ratio, total_occurrences = result
                else:
                    # Default values for new patterns
                    win_rate = 0.5
                    avg_gain_loss = 0.0
                    sharpe_ratio = 0.0
                    total_occurrences = 0
        else:
            # Get from memory storage
            key = f"{pattern_name}_{direction}"
            stats = self.memory_storage["pattern_stats"].get(key, {})
            win_rate = stats.get("win_rate", 0.5)
            avg_gain_loss = stats.get("avg_gain_loss", 0.0)
            sharpe_ratio = stats.get("sharpe_ratio", 0.0)
            total_occurrences = stats.get("total_occurrences", 0)
        
        # Calculate confidence score based on multiple factors
        # Base confidence starts at win rate
        confidence_score = win_rate
        
        # Adjust based on sample size (more occurrences = more confidence)
        if total_occurrences < 10:
            confidence_score *= 0.7  # Low sample size penalty
        elif total_occurrences < 30:
            confidence_score *= 0.85  # Medium sample size penalty
        
        # Adjust based on Sharpe ratio (risk-adjusted returns)
        if sharpe_ratio > 1.0:
            confidence_score *= 1.1  # Good risk-adjusted returns
        elif sharpe_ratio < 0:
            confidence_score *= 0.9  # Poor risk-adjusted returns
        
        # Cap confidence score between 0 and 1
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        return {
            "confidence_score": confidence_score,
            "win_rate": win_rate,
            "avg_gain_loss": avg_gain_loss,
            "sharpe_ratio": sharpe_ratio,
            "total_occurrences": total_occurrences
        }
    
    def _update_pattern_statistics(self):
        """Update pattern statistics based on outcomes."""
        if self.use_duckdb:
            with duckdb.connect(str(self.db_path)) as conn:
                # Calculate statistics for each pattern
                conn.execute("""
                    INSERT OR REPLACE INTO pattern_stats
                    SELECT 
                        pattern_name,
                        direction,
                        COUNT(*) as total_occurrences,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_outcomes,
                        AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as win_rate,
                        AVG((price_at_outcome - price_at_detection) / price_at_detection) as avg_gain_loss,
                        0.0 as sharpe_ratio,  -- TODO: Implement Sharpe ratio calculation
                        CURRENT_TIMESTAMP as last_updated
                    FROM pattern_outcomes
                    WHERE outcome_date IS NOT NULL
                    GROUP BY pattern_name, direction
                """)
        else:
            # Update in-memory statistics
            stats = {}
            for outcome in self.memory_storage["pattern_outcomes"]:
                if outcome["outcome_date"] is not None:
                    key = f"{outcome['pattern_name']}_{outcome['direction']}"
                    if key not in stats:
                        stats[key] = {
                            "total": 0,
                            "successful": 0,
                            "gains": []
                        }
                    
                    stats[key]["total"] += 1
                    if outcome["success"]:
                        stats[key]["successful"] += 1
                    
                    if outcome["price_at_outcome"] and outcome["price_at_detection"]:
                        gain = (outcome["price_at_outcome"] - outcome["price_at_detection"]) / outcome["price_at_detection"]
                        stats[key]["gains"].append(gain)
            
            # Calculate final statistics
            for key, data in stats.items():
                pattern_name, direction = key.split("_", 1)
                win_rate = data["successful"] / data["total"] if data["total"] > 0 else 0.5
                avg_gain_loss = sum(data["gains"]) / len(data["gains"]) if data["gains"] else 0.0
                
                self.memory_storage["pattern_stats"][key] = {
                    "pattern_name": pattern_name,
                    "direction": direction,
                    "total_occurrences": data["total"],
                    "successful_outcomes": data["successful"],
                    "win_rate": win_rate,
                    "avg_gain_loss": avg_gain_loss,
                    "sharpe_ratio": 0.0,  # TODO: Implement
                    "last_updated": datetime.now().isoformat()
                }
            
            self._save_memory_storage()
    
    def get_recent_performance(
        self,
        pattern_name: Optional[str] = None,
        days: int = 30
    ) -> List[Dict]:
        """
        Get recent pattern performance.
        
        Args:
            pattern_name: Filter by specific pattern (None for all)
            days: Number of days to look back
            
        Returns:
            List of recent pattern outcomes
        """
        since_date = datetime.now() - timedelta(days=days)
        
        if self.use_duckdb:
            with duckdb.connect(str(self.db_path)) as conn:
                query = """
                    SELECT * FROM pattern_outcomes
                    WHERE outcome_date > ?
                """
                params = [since_date]
                
                if pattern_name:
                    query += " AND pattern_name = ?"
                    params.append(pattern_name)
                
                query += " ORDER BY outcome_date DESC"
                
                results = conn.execute(query, params).fetchall()
                columns = [desc[0] for desc in conn.description]
                
                return [dict(zip(columns, row)) for row in results]
        else:
            # Filter from memory storage
            results = []
            for outcome in self.memory_storage["pattern_outcomes"]:
                if outcome["outcome_date"]:
                    outcome_date = datetime.fromisoformat(outcome["outcome_date"])
                    if outcome_date > since_date:
                        if pattern_name is None or outcome["pattern_name"] == pattern_name:
                            results.append(outcome)
            
            return sorted(results, key=lambda x: x["outcome_date"], reverse=True)


# Convenience functions for quick pattern outcome updates
def update_pattern_outcome(pattern_name: str, direction: str, success: bool):
    """
    Quick function to update the most recent pattern outcome.
    
    Args:
        pattern_name: Name of the pattern
        direction: Pattern direction ('bullish' or 'bearish')
        success: Whether the pattern was successful
    """
    engine = PatternConfidenceEngine()
    
    # Find the most recent detection without an outcome
    if engine.use_duckdb:
        with duckdb.connect(str(engine.db_path)) as conn:
            result = conn.execute("""
                SELECT id, price_at_detection
                FROM pattern_outcomes
                WHERE pattern_name = ? AND direction = ? AND outcome_date IS NULL
                ORDER BY detected_date DESC
                LIMIT 1
            """, [pattern_name, direction]).fetchone()
            
            if result:
                detection_id, price_at_detection = result
                # For demo purposes, assume 2% gain/loss
                price_change = 0.02 if success else -0.02
                price_at_outcome = price_at_detection * (1 + price_change)
                
                engine.update_pattern_outcome(
                    detection_id,
                    success,
                    price_at_outcome
                )
    else:
        # Find in memory storage
        latest_detection = None
        for outcome in reversed(engine.memory_storage["pattern_outcomes"]):
            if (outcome["pattern_name"] == pattern_name and 
                outcome["direction"] == direction and 
                outcome["outcome_date"] is None):
                latest_detection = outcome
                break
        
        if latest_detection:
            # For demo purposes, assume 2% gain/loss
            price_change = 0.02 if success else -0.02
            price_at_outcome = latest_detection["price_at_detection"] * (1 + price_change)
            
            engine.update_pattern_outcome(
                latest_detection["id"],
                success,
                price_at_outcome
            ) 