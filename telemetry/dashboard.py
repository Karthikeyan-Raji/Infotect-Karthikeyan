# ==============================================================================
# ChronosMatch: Zero-Copy High-Frequency Trading Engine
# Module: telemetry/dashboard.py
# Role: Person 2 (Ingestion, UI & Auditing Architect)
# Description: Real-time Terminal User Interface (TUI) rendering Best Bid/Offer (BBO),
#              Level 2 Depth Ladder with dynamic visual volume bars, rolling p50/p99
#              latency histograms, and institutional "Whale" order alerts.
# ==============================================================================

import time
import sys
import os
from typing import Dict, Any, List

# Check curses availability with graceful ANSI fallback
try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False

class TelemetryDashboard:
    """
    High-Frequency Terminal Dashboard for ChronosMatch.
    Renders sub-millisecond market microstructure updates without blocking engine threads.
    """
    def __init__(self, use_curses: bool = True):
        self.use_curses = use_curses and CURSES_AVAILABLE
        self.stdscr = None
        self.whale_alerts: List[Dict[str, Any]] = []
        self.max_whale_alerts = 6
        self.last_render_time = 0.0

    def start(self):
        if self.use_curses:
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            curses.curs_set(0)
            if curses.has_colors():
                curses.start_color()
                curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Bids
                curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)    # Asks
                curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Whales
                curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Header
                curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)   # Banner
            self.stdscr.nodelay(True)

    def stop(self):
        if self.use_curses and self.stdscr is not None:
            curses.nocbreak()
            self.stdscr.keypad(False)
            curses.echo()
            curses.curs_set(1)
            curses.endwin()
            self.stdscr = None

    def push_whale_alert(self, order_id: int, side: str, price: float, qty: int):
        alert = {
            "time": time.strftime("%H:%M:%S"),
            "order_id": order_id,
            "side": side,
            "price": price,
            "qty": qty
        }
        self.whale_alerts.insert(0, alert)
        if len(self.whale_alerts) > self.max_whale_alerts:
            self.whale_alerts.pop()

    def render(self, market_state: Dict[str, Any], latency_stats: Dict[str, Any], throughput_ops: float):
        """Renders the complete telemetry dashboard."""
        now = time.perf_counter()
        # Cap TUI refresh rate to 30 FPS to prevent terminal rendering overhead
        if now - self.last_render_time < 0.033:
            return
        self.last_render_time = now

        if self.use_curses and self.stdscr is not None:
            self._render_curses(market_state, latency_stats, throughput_ops)
        else:
            self._render_ansi(market_state, latency_stats, throughput_ops)

    def _render_curses(self, state: Dict[str, Any], l_stats: Dict[str, Any], t_rate: float):
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()
        if h < 24 or w < 80:
            self.stdscr.addstr(0, 0, "Terminal window too small. Expand to at least 80x24.")
            self.stdscr.refresh()
            return

        # Header Banner
        banner = " CHRONOSMATCH :: ZERO-COPY HIGH-FREQUENCY MATCHING ENGINE "
        self.stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
        self.stdscr.addstr(0, 0, banner.center(w - 1))
        self.stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)

        # Telemetry & Throughput Section
        spread_bps = (state["spread"] / state["best_ask"] * 10_000) if state.get("best_ask", 0) > 0 else 0.0
        self.stdscr.addstr(2, 2, f"Target SLA: < 50.0 µs  |  Throughput: {t_rate:,.0f} ticks/sec", curses.A_BOLD)
        self.stdscr.addstr(3, 2, f"Total Orders: {state['total_orders']:,}  |  Matches: {state['total_fills']:,}  |  Volume: {state['total_volume']:,}")

        # Latency Box
        self.stdscr.attron(curses.color_pair(4))
        self.stdscr.addstr(5, 2, "┌─ ENGINE LATENCY PROFILE (Microseconds) ─────────────────────────┐")
        self.stdscr.addstr(6, 2, f"│  p50: {l_stats.get('p50_us', 0.0):6.2f} µs   p90: {l_stats.get('p90_us', 0.0):6.2f} µs   p99: {l_stats.get('p99_us', 0.0):6.2f} µs   │")
        self.stdscr.addstr(7, 2, f"│  p99.9: {l_stats.get('p99_9_us', 0.0):6.2f} µs  Max: {l_stats.get('max_us', 0.0):6.2f} µs   Mean: {l_stats.get('mean_us', 0.0):6.2f} µs  │")
        self.stdscr.addstr(8, 2, "└─────────────────────────────────────────────────────────────────┘")
        self.stdscr.attroff(curses.color_pair(4))

        # Best Bid / Offer Banner
        bbo_str = f"BBO SPREAD: ${state['spread']:.2f} ({spread_bps:.1f} bps)  |  Best Bid: ${state['best_bid']:.2f}  |  Best Ask: ${state['best_ask']:.2f}"
        self.stdscr.addstr(10, 2, bbo_str, curses.A_UNDERLINE | curses.A_BOLD)

        # Depth Ladder (Top 5 Bids vs Top 5 Asks)
        self.stdscr.addstr(12, 2, "── TOP BIDS (BUY) ─────────────┼── TOP ASKS (SELL) ────────────")
        bids = state.get("bids", [])
        asks = state.get("asks", [])

        for i in range(5):
            line_y = 13 + i
            bid_str = "    --        --     "
            if i < len(bids):
                b_px, b_vol = bids[i]
                bar = "█" * min(10, int(b_vol / 200) + 1)
                bid_str = f"${b_px:6.2f}  {b_vol:5d}  {bar:<10}"

            ask_str = "    --        --     "
            if i < len(asks):
                a_px, a_vol = asks[i]
                bar = "█" * min(10, int(a_vol / 200) + 1)
                ask_str = f"${a_px:6.2f}  {a_vol:5d}  {bar:<10}"

            self.stdscr.attron(curses.color_pair(1))
            self.stdscr.addstr(line_y, 2, bid_str)
            self.stdscr.attroff(curses.color_pair(1))

            self.stdscr.addstr(line_y, 31, "│")

            self.stdscr.attron(curses.color_pair(2))
            self.stdscr.addstr(line_y, 34, ask_str)
            self.stdscr.attroff(curses.color_pair(2))

        # Institutional Whale Alert Feed
        self.stdscr.attron(curses.color_pair(3) | curses.A_BOLD)
        self.stdscr.addstr(19, 2, "▲ INSTITUTIONAL WHALE ORDER ALERTS (Qty >= 5,000) ▲")
        self.stdscr.attroff(curses.color_pair(3) | curses.A_BOLD)

        for idx, alert in enumerate(self.whale_alerts):
            if 20 + idx >= h - 1:
                break
            side_color = curses.color_pair(1) if alert["side"] == 'B' else curses.color_pair(2)
            self.stdscr.attron(side_color)
            alert_line = f"[{alert['time']}] BLOCK {alert['side']} ORD#{alert['order_id']} :: {alert['qty']:,} shares @ ${alert['price']:.2f}"
            self.stdscr.addstr(20 + idx, 2, alert_line)
            self.stdscr.attroff(side_color)

        self.stdscr.refresh()

    def _render_ansi(self, state: Dict[str, Any], l_stats: Dict[str, Any], t_rate: float):
        """ANSI escape code fallback for terminals without python-curses."""
        spread_bps = (state["spread"] / state["best_ask"] * 10_000) if state.get("best_ask", 0) > 0 else 0.0
        output = [
            "\033[2J\033[H",  # Clear screen and cursor home
            "=" * 75,
            " CHRONOSMATCH :: ZERO-COPY HIGH-FREQUENCY MATCHING ENGINE",
            "=" * 75,
            f"Throughput: {t_rate:,.0f} ticks/sec | Target SLA: < 50 µs",
            f"Total Orders: {state['total_orders']:,} | Matches: {state['total_fills']:,} | Volume: {state['total_volume']:,}",
            "-" * 75,
            f"LATENCY p50: {l_stats.get('p50_us', 0.0):.2f} µs | p90: {l_stats.get('p90_us', 0.0):.2f} µs | p99: {l_stats.get('p99_us', 0.0):.2f} µs | Max: {l_stats.get('max_us', 0.0):.2f} µs",
            f"BBO SPREAD: ${state['spread']:.2f} ({spread_bps:.1f} bps) [Bid: ${state['best_bid']:.2f} | Ask: ${state['best_ask']:.2f}]",
            "-" * 75,
            "TOP BIDS (BUY)                   | TOP ASKS (SELL)"
        ]

        bids = state.get("bids", [])
        asks = state.get("asks", [])
        for i in range(5):
            b_s = f"${bids[i][0]:.2f} ({bids[i][1]} shs)" if i < len(bids) else "----------------"
            a_s = f"${asks[i][0]:.2f} ({asks[i][1]} shs)" if i < len(asks) else "----------------"
            output.append(f"{b_s:<32} | {a_s}")

        output.append("-" * 75)
        output.append("INSTITUTIONAL WHALE ORDER ALERTS:")
        for a in self.whale_alerts[:3]:
            output.append(f"  [{a['time']}] BLOCK {a['side']} ORD#{a['order_id']} -> {a['qty']:,} @ ${a['price']:.2f}")
        output.append("=" * 75)

        sys.stdout.write("\n".join(output) + "\n")
        sys.stdout.flush()
