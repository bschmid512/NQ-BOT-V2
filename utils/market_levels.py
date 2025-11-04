"""
Market Levels Module - Critical NQ Levels
Tracks PDH, PDL, PDC, overnight range, and key reference levels
"""
import pandas as pd
from datetime import datetime, timedelta, time
import pytz
from typing import Dict, Optional


class MarketLevels:
    """
    Track critical price levels for NQ trading
    Professional traders know these by heart
    """
    
    def __init__(self):
        self.levels = {}
        self.overnight_high = None
        self.overnight_low = None
        self.daily_open = None
        
    def update_levels(self, df: pd.DataFrame) -> Dict:
        """
        Calculate all critical levels
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with all levels
        """
        if df.empty:
            return self.levels
        
        current_date = df.index[-1].date()
        
        # Get previous trading day
        prev_day = current_date - timedelta(days=1)
        while prev_day.weekday() > 4:  # Skip weekends
            prev_day -= timedelta(days=1)
        
        # Previous Day Levels
        prev_day_data = df[df.index.date == prev_day]
        if not prev_day_data.empty:
            self.levels['pdh'] = prev_day_data['high'].max()
            self.levels['pdl'] = prev_day_data['low'].min()
            self.levels['pdc'] = prev_day_data['close'].iloc[-1]
        
        # Previous Week Levels
        prev_week_start = current_date - timedelta(days=7)
        prev_week_data = df[(df.index.date >= prev_week_start.date()) & 
                           (df.index.date < current_date)]
        if not prev_week_data.empty:
            self.levels['pwh'] = prev_week_data['high'].max()
            self.levels['pwl'] = prev_week_data['low'].min()
        
        # Overnight Range (6pm - 9:30am ET)
        et_tz = pytz.timezone('US/Eastern')
        today_930am = et_tz.localize(datetime.combine(current_date, time(9, 30)))
        yesterday_6pm = et_tz.localize(datetime.combine(prev_day, time(18, 0)))
        
        overnight_data = df[(df.index >= yesterday_6pm) & (df.index < today_930am)]
        if not overnight_data.empty:
            self.overnight_high = overnight_data['high'].max()
            self.overnight_low = overnight_data['low'].min()
            self.levels['on_high'] = self.overnight_high
            self.levels['on_low'] = self.overnight_low
        
        # Today's Open (9:30am ET price)
        today_data = df[df.index.date == current_date]
        if not today_data.empty:
            self.daily_open = today_data['open'].iloc[0]
            self.levels['daily_open'] = self.daily_open
        
        # Gap Analysis
        if 'pdc' in self.levels and self.daily_open:
            self.levels['gap_size'] = self.daily_open - self.levels['pdc']
            self.levels['gap_pct'] = self.levels['gap_size'] / self.levels['pdc']
            
            # Classify gap
            if abs(self.levels['gap_pct']) < 0.001:
                self.levels['gap_type'] = 'NO_GAP'
            elif self.levels['gap_size'] > 0:
                if self.levels['gap_pct'] > 0.01:
                    self.levels['gap_type'] = 'LARGE_GAP_UP'
                else:
                    self.levels['gap_type'] = 'SMALL_GAP_UP'
            else:
                if self.levels['gap_pct'] < -0.01:
                    self.levels['gap_type'] = 'LARGE_GAP_DOWN'
                else:
                    self.levels['gap_type'] = 'SMALL_GAP_DOWN'
        
        # Round number levels
        current_price = df['close'].iloc[-1]
        self.levels['round_numbers'] = self._get_round_numbers(current_price)
        
        # Monthly open
        month_start = current_date.replace(day=1)
        month_data = df[df.index.date >= month_start]
        if not month_data.empty:
            self.levels['monthly_open'] = month_data['open'].iloc[0]
        
        return self.levels
    
    def _get_round_numbers(self, price: float) -> Dict:
        """
        Get nearby round number levels
        These act as psychological support/resistance
        """
        levels = {}
        
        # Whole handles (500 point intervals)
        whole = round(price / 500) * 500
        levels['whole_handle'] = whole
        levels['whole_above'] = whole + 500
        levels['whole_below'] = whole - 500
        
        # Half handles (250 point intervals)
        half = round(price / 250) * 250
        levels['half_handle'] = half
        
        # Quarter handles (125 point intervals)
        quarter = round(price / 125) * 125
        levels['quarter_handle'] = quarter
        
        # Identify which level is closest
        distances = {
            'whole': abs(price - whole),
            'half': abs(price - half),
            'quarter': abs(price - quarter)
        }
        levels['nearest_level_type'] = min(distances, key=distances.get)
        levels['nearest_level_distance'] = min(distances.values())
        
        return levels
    
    def get_context(self, current_price: float) -> Dict:
        """
        Get market context relative to key levels
        This is what you need to know before taking a trade
        """
        context = {}
        
        if not self.levels:
            return context
        
        # Position relative to previous day
        if 'pdh' in self.levels and 'pdl' in self.levels:
            context['above_pdh'] = current_price > self.levels['pdh']
            context['below_pdl'] = current_price < self.levels['pdl']
            context['inside_prev_day_range'] = (
                self.levels['pdl'] < current_price < self.levels['pdh']
            )
            
            # Distance to PDH/PDL in points
            context['dist_to_pdh'] = self.levels['pdh'] - current_price
            context['dist_to_pdl'] = current_price - self.levels['pdl']
        
        # Position relative to overnight range
        if self.overnight_high and self.overnight_low:
            context['above_on_high'] = current_price > self.overnight_high
            context['below_on_low'] = current_price < self.overnight_low
            context['inside_overnight_range'] = (
                self.overnight_low < current_price < self.overnight_high
            )
        
        # Gap status
        if 'gap_type' in self.levels and 'gap_size' in self.levels:
            context['gap_type'] = self.levels['gap_type']
            context['gap_size'] = self.levels['gap_size']
            
            # Gap fill progress
            if self.levels['gap_type'] != 'NO_GAP':
                if self.levels['gap_size'] > 0:  # Gap up
                    gap_fill_pct = max(0, (self.levels['daily_open'] - current_price) / 
                                     self.levels['gap_size'])
                else:  # Gap down
                    gap_fill_pct = max(0, (current_price - self.levels['daily_open']) / 
                                     abs(self.levels['gap_size']))
                context['gap_fill_pct'] = gap_fill_pct
        
        # Round number proximity
        if 'round_numbers' in self.levels:
            rn = self.levels['round_numbers']
            context['near_round_number'] = rn['nearest_level_distance'] < 20
            context['nearest_round_level'] = rn[rn['nearest_level_type'] + '_handle']
        
        return context
    
    def is_near_major_level(self, price: float, threshold: float = 15) -> bool:
        """
        Check if price is near a major level (within threshold points)
        Use this to avoid entries near resistance/support
        """
        if not self.levels:
            return False
        
        major_levels = []
        
        # Add all major levels
        for key in ['pdh', 'pdl', 'pdc', 'pwh', 'pwl', 'on_high', 'on_low', 
                    'daily_open', 'monthly_open']:
            if key in self.levels:
                major_levels.append(self.levels[key])
        
        # Check proximity
        for level in major_levels:
            if abs(price - level) < threshold:
                return True
        
        return False
    
    def get_nearest_level(self, price: float) -> tuple[str, float]:
        """
        Find the nearest major level and its distance
        Returns: (level_name, distance_in_points)
        """
        if not self.levels:
            return None, None
        
        distances = {}
        
        for key in ['pdh', 'pdl', 'pdc', 'on_high', 'on_low', 'daily_open']:
            if key in self.levels:
                distances[key] = abs(price - self.levels[key])
        
        if not distances:
            return None, None
        
        nearest = min(distances, key=distances.get)
        return nearest, distances[nearest]
    
    def print_levels(self):
        """Print all levels for debugging"""
        print("\n" + "=" * 60)
        print("MARKET LEVELS")
        print("=" * 60)
        
        if 'pdh' in self.levels:
            print(f"Previous Day High:    {self.levels['pdh']:.2f}")
        if 'pdl' in self.levels:
            print(f"Previous Day Low:     {self.levels['pdl']:.2f}")
        if 'pdc' in self.levels:
            print(f"Previous Day Close:   {self.levels['pdc']:.2f}")
        
        if 'on_high' in self.levels:
            print(f"\nOvernight High:       {self.levels['on_high']:.2f}")
        if 'on_low' in self.levels:
            print(f"Overnight Low:        {self.levels['on_low']:.2f}")
        
        if 'daily_open' in self.levels:
            print(f"\nToday's Open:         {self.levels['daily_open']:.2f}")
        
        if 'gap_type' in self.levels:
            print(f"\nGap Type:             {self.levels['gap_type']}")
            print(f"Gap Size:             {self.levels['gap_size']:.2f} points")
        
        if 'round_numbers' in self.levels:
            rn = self.levels['round_numbers']
            print(f"\nNearest Round Number: {rn[rn['nearest_level_type'] + '_handle']:.2f}")
            print(f"Distance:             {rn['nearest_level_distance']:.2f} points")
        
        print("=" * 60 + "\n")


# Global instance
market_levels = MarketLevels()
