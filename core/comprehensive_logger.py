"""
Comprehensive Logging System
Tracks all decisions, missed opportunities, and system performance
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from config import LOG_DIR


class ComprehensiveLogger:
    """
    Logs EVERYTHING for post-analysis
    - Why trades were taken
    - Why trades were rejected
    - What vision saw vs what actually happened
    - Missed opportunities
    """
    
    def __init__(self):
        # Create log files
        self.decision_log_file = LOG_DIR / 'decisions.jsonl'
        self.opportunity_log_file = LOG_DIR / 'missed_opportunities.jsonl'
        self.fusion_log_file = LOG_DIR / 'fusion_decisions.jsonl'
        self.vision_log_file = LOG_DIR / 'vision_analysis.jsonl'
        
        print(f"✅ Comprehensive Logger initialized")
        print(f"   Decision log: {self.decision_log_file}")
        print(f"   Opportunity log: {self.opportunity_log_file}")
    
    def log_decision(self, decision_type: str, details: Dict):
        """
        Log any decision made by the system
        
        Args:
            decision_type: 'trade_taken', 'trade_rejected', 'position_closed', etc.
            details: Dictionary with all relevant information
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': decision_type,
            'details': details
        }
        
        self._write_json_line(self.decision_log_file, log_entry)
    
    def log_trade_taken(self, signal: Dict, position: Dict, context: Dict):
        """Log when a trade is taken"""
        self.log_decision('trade_taken', {
            'signal': signal,
            'position': position,
            'context': {
                'market_regime': context.get('market_regime'),
                'momentum': context.get('momentum'),
                'vision_confirmed': context.get('vision_available'),
                'vision_sentiment': context.get('vision_sentiment')
            }
        })
        
        print(f"📝 TRADE TAKEN logged: {signal['direction']} @ {signal.get('entry', signal.get('price'))}")
    
    def log_trade_rejected(self, signals: List[Dict], reason: str, context: Dict):
        """Log when signals were generated but trade was rejected"""
        self.log_decision('trade_rejected', {
            'signals': signals,
            'rejection_reason': reason,
            'context': context
        })
        
        print(f"⛔ TRADE REJECTED logged: {reason}")
    
    def log_fusion_decision(self, signals: List[Dict], fusion_result: Optional[Dict],
                           context: Dict):
        """Log fusion engine decision"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'input_signals': signals,
            'fusion_approved': fusion_result is not None,
            'fused_signal': fusion_result,
            'context': context
        }
        
        self._write_json_line(self.fusion_log_file, entry)
    
    def log_vision_analysis(self, vision_data: Dict, price_action: Dict):
        """
        Log what vision system saw and compare to price action
        This helps analyze if vision is accurate
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'vision_sentiment': vision_data.get('statistics', {}).get('sentiment'),
            'vision_patterns': vision_data.get('patterns', []),
            'visual_levels': vision_data.get('support_resistance', {}),
            'actual_price': price_action.get('close'),
            'actual_direction': price_action.get('direction'),
            'price_change': price_action.get('change_pct')
        }
        
        self._write_json_line(self.vision_log_file, entry)
    
    def log_missed_opportunity(self, opportunity_type: str, details: Dict):
        """
        Log potential opportunities that were missed
        
        Args:
            opportunity_type: 'missed_rally', 'missed_breakdown', 'late_entry', etc.
            details: What happened and why we missed it
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'opportunity_type': opportunity_type,
            'details': details
        }
        
        self._write_json_line(self.opportunity_log_file, entry)
        
        print(f"💡 OPPORTUNITY logged: {opportunity_type}")
    
    def log_position_update(self, position_id: int, update_type: str, details: Dict):
        """Log position updates (stop hit, target reached, etc.)"""
        self.log_decision('position_update', {
            'position_id': position_id,
            'update_type': update_type,
            'details': details
        })
    
    def analyze_missed_opportunities(self, lookback_hours: int = 24) -> Dict:
        """
        Analyze recent missed opportunities
        Returns summary of what we could have caught
        """
        opportunities = self._read_recent_logs(
            self.opportunity_log_file, 
            lookback_hours
        )
        
        summary = {
            'total_opportunities': len(opportunities),
            'by_type': {},
            'estimated_points_missed': 0
        }
        
        for opp in opportunities:
            opp_type = opp.get('opportunity_type', 'unknown')
            summary['by_type'][opp_type] = summary['by_type'].get(opp_type, 0) + 1
            
            # Try to estimate points missed
            if 'move_size' in opp.get('details', {}):
                summary['estimated_points_missed'] += opp['details']['move_size']
        
        return summary
    
    def analyze_fusion_performance(self, lookback_hours: int = 24) -> Dict:
        """
        Analyze fusion engine performance
        - Approval rate
        - Success rate of approved signals
        """
        decisions = self._read_recent_logs(self.fusion_log_file, lookback_hours)
        
        total = len(decisions)
        approved = len([d for d in decisions if d.get('fusion_approved')])
        
        return {
            'total_evaluations': total,
            'approved': approved,
            'approval_rate': approved / total if total > 0 else 0,
            'rejection_rate': (total - approved) / total if total > 0 else 0
        }
    
    def get_recent_decisions(self, limit: int = 50) -> List[Dict]:
        """Get recent decisions for dashboard display"""
        return self._read_recent_logs(self.decision_log_file, hours=24, limit=limit)
    
    def _write_json_line(self, file_path: Path, data: Dict):
        """Write a JSON line to log file"""
        try:
            with open(file_path, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            print(f"❌ Error writing to log: {e}")
    
    def _read_recent_logs(self, file_path: Path, hours: int = 24, 
                         limit: Optional[int] = None) -> List[Dict]:
        """Read recent log entries"""
        if not file_path.exists():
            return []
        
        try:
            entries = []
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        entry_time = datetime.fromisoformat(entry['timestamp']).timestamp()
                        
                        if entry_time >= cutoff_time:
                            entries.append(entry)
                    except:
                        continue
            
            # Return most recent first
            entries.reverse()
            
            if limit:
                return entries[:limit]
            return entries
            
        except Exception as e:
            print(f"❌ Error reading log: {e}")
            return []
    
    def generate_daily_report(self) -> str:
        """Generate daily performance report"""
        fusion_perf = self.analyze_fusion_performance(24)
        missed_opps = self.analyze_missed_opportunities(24)
        recent_decisions = self.get_recent_decisions(20)
        
        report = f"""
{'='*70}
DAILY TRADING REPORT - {datetime.now().strftime('%Y-%m-%d')}
{'='*70}

FUSION ENGINE PERFORMANCE:
  Total Evaluations: {fusion_perf['total_evaluations']}
  Signals Approved:  {fusion_perf['approved']} ({fusion_perf['approval_rate']:.1%})
  Signals Rejected:  {fusion_perf['total_evaluations'] - fusion_perf['approved']} ({fusion_perf['rejection_rate']:.1%})

MISSED OPPORTUNITIES:
  Total Opportunities Identified: {missed_opps['total_opportunities']}
  Estimated Points Missed: {missed_opps['estimated_points_missed']:.0f}
  By Type: {missed_opps['by_type']}

RECENT DECISIONS (Last 20):
"""
        for i, decision in enumerate(recent_decisions[:20], 1):
            report += f"  {i}. {decision['type']}: {decision['timestamp']}\n"
        
        report += f"\n{'='*70}\n"
        
        return report


# Create global instance
comprehensive_logger = ComprehensiveLogger()
