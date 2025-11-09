"""
AI-Powered Channel Analytics Agent
Provides deep analysis of investment channels (brokers/platforms) using GPT-5
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from assistant_helper import get_shared_assistant  # type: ignore
from .base_agent import BaseAgent


class AIChannelAnalyticsAgent(BaseAgent):
    """
    AI-powered agent for intelligent channel analytics
    """

    def __init__(self, agent_id: str = "ai_channel_agent") -> None:
        super().__init__(agent_id, "AI Channel Analytics Agent")
        self.capabilities = [
            "channel_distribution_analysis",
            "channel_performance_benchmarking",
            "cost_efficiency_analysis",
            "execution_quality_monitoring",
            "channel_risk_scoring",
            "platform_recommendations",
        ]

        self.channel_cache: List[Dict[str, Any]] = []
        self.last_analysis_data: Dict[str, Any] = {}
        self.channel_summary: Dict[str, Any] = {}

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis method using AI for channel analytics
        """
        try:
            self.update_status("analyzing")
            self.last_analysis_data = data

            insights = self._ai_analyze_channels(data)
            self.channel_cache = insights

            self.update_status("active")
            response = self.format_response(insights, "high")
            if self.channel_summary:
                response["metadata"] = {"channel_summary": self.channel_summary}
            return response

        except Exception as exc:
            self.logger.error(f"Error in AI channel analytics: {exc}")
            self.update_status("error")
            return self.format_response([], "low", error=str(exc))

    def _ai_analyze_channels(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use AI to intelligently analyze channel data"""

        portfolio_data = data.get("portfolio_data", {})
        holdings = portfolio_data.get("holdings") or data.get("holdings") or []
        transactions = data.get("transactions") or []
        user_profile = data.get("user_profile") or {}

        if not holdings:
            return [
                {
                    "type": "channel_analysis",
                    "severity": "low",
                    "title": "No Channel Data",
                    "description": "No holdings with channel information were found.",
                    "recommendation": "Upload transactions with channel/platform metadata to enable channel analytics.",
                    "data": {"channel_data_available": False},
                }
            ]

        # Prepare structured summary for AI prompt and cached metadata
        self.channel_summary = self._prepare_channel_summary(holdings, transactions)

        # Create AI prompt for channel analysis
        prompt = self._create_channel_analysis_prompt(self.channel_summary, user_profile)

        try:
            current_date = datetime.now().strftime("%Y-%m-%d")

            runner = get_shared_assistant()
            ai_response = runner.run(
                user_message=(
                    f"Current date: {current_date}. Analyze the following channel metrics and return JSON insights as specified.\n\n{prompt}"
                ),
                function_map={},
                extra_messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a wealth management channel specialist. Provide platform-level insights, "
                            "compare execution quality, flag underperforming brokers, and recommend optimization actions."
                        ),
                    }
                ],
            )

            if not ai_response or not ai_response.strip():
                self.logger.error("AI response content is empty for channel analytics")
                return []

            insights = self._parse_ai_channel_response(ai_response)
            if insights:
                return insights

            return [
                {
                    "type": "channel_analysis",
                    "severity": "medium",
                    "title": "Channel Analytics Parsing Error",
                    "description": "Channel analytics agent could not parse AI output. Review channel summary manually.",
                    "recommendation": "Check the channel summary metadata for raw metrics.",
                    "data": {"channel_summary": self.channel_summary},
                }
            ]

        except Exception as exc:
            self.logger.error(f"AI channel analytics failed: {exc}")
            return [
                {
                    "type": "channel_analysis",
                    "severity": "medium",
                    "title": "Channel Analytics Error",
                    "description": f"AI channel analytics failed: {exc}",
                    "recommendation": "Please try again or contact support.",
                    "data": {"error": str(exc)},
                }
            ]

    def _prepare_channel_summary(
        self, holdings: List[Dict[str, Any]], transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare structured channel summary for AI analysis"""

        df = pd.DataFrame(holdings)
        if "channel" not in df.columns:
            df["channel"] = "Unknown"
        df["channel"].fillna("Unknown", inplace=True)

        # Determine key numeric columns
        value_columns = [col for col in df.columns if "current_value" in col.lower() or "value" in col.lower()]
        investment_columns = [col for col in df.columns if "investment" in col.lower() or "cost" in col.lower()]
        pnl_columns = [col for col in df.columns if "pnl" in col.lower() or "profit" in col.lower()]
        quantity_columns = [col for col in df.columns if "total_quantity" in col.lower() or "quantity" in col.lower()]
        price_columns = [col for col in df.columns if "current_price" in col.lower() or "price" in col.lower()]

        # Fallback calculations if values missing
        if not value_columns and quantity_columns:
            qty_col = quantity_columns[0]
            price_col = price_columns[0] if price_columns else None
            if price_col:
                qty_series = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
                price_series = pd.to_numeric(df[price_col], errors="coerce").fillna(0)
                df["_computed_current_value"] = qty_series * price_series
                value_columns.append("_computed_current_value")

        if not investment_columns and quantity_columns:
            qty_col = quantity_columns[0]
            avg_price_col = next((col for col in df.columns if "average_price" in col.lower()), None)
            if avg_price_col:
                qty_series = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
                avg_price_series = pd.to_numeric(df[avg_price_col], errors="coerce").fillna(0)
                df["_computed_investment"] = qty_series * avg_price_series
                investment_columns.append("_computed_investment")

        if not pnl_columns:
            if value_columns and investment_columns:
                df[value_columns[0]] = pd.to_numeric(df[value_columns[0]], errors="coerce").fillna(0)
                df[investment_columns[0]] = pd.to_numeric(df[investment_columns[0]], errors="coerce").fillna(0)
                df["_computed_pnl"] = df[value_columns[0]] - df[investment_columns[0]]
                pnl_columns.append("_computed_pnl")

        value_col = value_columns[0] if value_columns else None
        investment_col = investment_columns[0] if investment_columns else None
        pnl_col = pnl_columns[0] if pnl_columns else None

        for numeric_col in {value_col, investment_col, pnl_col}:
            if numeric_col and numeric_col in df.columns:
                df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce").fillna(0)
        if quantity_columns:
            qty_col = quantity_columns[0]
            df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)

        # Channel aggregates
        channel_group = df.groupby("channel")
        channel_stats = []

        for channel, group in channel_group:
            total_value = float(group[value_col].sum()) if value_col is not None else None
            total_investment = float(group[investment_col].sum()) if investment_col is not None else None
            pnl_value = float(group[pnl_col].sum()) if pnl_col is not None else None

            avg_trade_size = None
            if quantity_columns and investment_col is not None:
                quant_series = pd.to_numeric(group[quantity_columns[0]], errors="coerce").fillna(0)
                if not quant_series.empty and quant_series.sum() > 0:
                    avg_trade_size = float(group[investment_col].sum() / quant_series.sum())

            channel_stats.append(
                {
                    "channel": channel,
                    "holdings_count": int(len(group)),
                    "unique_tickers": int(group["ticker"].nunique()) if "ticker" in group.columns else None,
                    "total_investment": total_investment,
                    "current_value": total_value,
                    "total_pnl": pnl_value,
                    "pnl_percentage": (pnl_value / total_investment * 100) if total_investment else None,
                    "average_trade_value": avg_trade_size,
                }
            )

        channel_stats_sorted = sorted(
            channel_stats,
            key=lambda entry: entry.get("total_investment") or 0,
            reverse=True,
        )

        # Transaction level metrics by channel (fills, frequency)
        channel_trades: Dict[str, Dict[str, Any]] = {}
        if transactions:
            df_txn = pd.DataFrame(transactions)
            if "channel" not in df_txn.columns:
                df_txn["channel"] = "Unknown"
            df_txn["channel"].fillna("Unknown", inplace=True)
            df_txn["date"] = pd.to_datetime(df_txn.get("date"), errors="coerce")

            txn_group = df_txn.groupby("channel")
            for channel, group in txn_group:
                channel_trades[channel] = {
                    "transaction_count": int(len(group)),
                    "buy_count": int(group[group.get("transaction_type") == "buy"].shape[0])
                    if "transaction_type" in group.columns
                    else None,
                    "sell_count": int(group[group.get("transaction_type") == "sell"].shape[0])
                    if "transaction_type" in group.columns
                    else None,
                    "first_transaction": group["date"].min().strftime("%Y-%m-%d") if "date" in group else None,
                    "last_transaction": group["date"].max().strftime("%Y-%m-%d") if "date" in group else None,
                    "avg_ticket_size": float(
                        pd.to_numeric(group["price"], errors="coerce").dropna().mean()
                    )
                    if "price" in group.columns and not pd.to_numeric(group["price"], errors="coerce").dropna().empty
                    else None,
                }

        overview = {
            "channels_detected": [stat["channel"] for stat in channel_stats_sorted],
            "channel_count": len(channel_stats_sorted),
            "total_holdings_analyzed": int(len(df)),
            "total_value_across_channels": float(df[value_col].sum()) if value_col is not None else None,
            "total_investment_across_channels": float(df[investment_col].sum()) if investment_col is not None else None,
            "total_pnl_across_channels": float(df[pnl_col].sum()) if pnl_col is not None else None,
        }

        best_channel = next(
            (stat for stat in channel_stats_sorted if stat.get("pnl_percentage") is not None),
            None,
        )
        worst_channel = next(
            (stat for stat in reversed(channel_stats_sorted) if stat.get("pnl_percentage") is not None),
            None,
        )

        summary = {
            "overview": overview,
            "channel_statistics": channel_stats_sorted,
            "transactions_by_channel": channel_trades,
            "best_channel": best_channel,
            "worst_channel": worst_channel,
        }

        return summary

    def _create_channel_analysis_prompt(
        self, channel_summary: Dict[str, Any], user_profile: Dict[str, Any]
    ) -> str:
        """Create AI prompt for channel analytics"""

        return f"""
Analyze investment channel performance using the data provided below. Focus on broker/platform quality, execution efficiency, cost leakage, and platform concentration.

USER PROFILE:
{json.dumps(user_profile, indent=2, default=str)}

CHANNEL SUMMARY:
{json.dumps(channel_summary, indent=2, default=str)}

Return ONLY a JSON object with the following structure:
{{
  "insights": [
    {{
      "type": "channel_performance",
      "severity": "high" | "medium" | "low",
      "title": "Short descriptive title",
      "description": "Detailed narrative insight (concise paragraphs)",
      "recommendation": "Actionable next step for the investor",
      "data": {{
        "channel": "Channel name",
        "metrics": {{
          "current_value": number,
          "total_investment": number,
          "pnl_percentage": number
        }},
        "tags": ["best_execution", "high_costs", "underutilized", ...]
      }}
    }}
  ],
  "channel_watchlist": [
    {{
      "channel": "Channel name",
      "issue": "Key issue (liquidity, costs, operational risks, etc.)",
      "urgency": "high" | "medium" | "low",
      "suggested_action": "Immediate action to take",
      "supporting_metrics": {{"metric_name": value}}
    }}
  ],
  "channel_recommendations": {{
    "consolidation_opportunities": ["Channel suggestions"],
    "upgrade_opportunities": ["Channel suggestions"],
    "monitoring_metrics": ["Metric names to monitor routinely"]
  }}
}}

CRITICAL REQUIREMENTS:
- Return valid JSON only. No markdown, code fences, or commentary.
- Use INR amounts where available and percentages with two decimals.
- Highlight channels that significantly outperform or underperform.
- Identify diversification or consolidation opportunities across channels.
- Suggest monitoring cadences (e.g., monthly execution review, quarterly platform audit).
"""

    def _parse_ai_channel_response(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI response and extract channel insights"""
        cleaned = ai_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            self.logger.error("Channel analytics JSON parsing failed")
            return []

        insights: List[Dict[str, Any]] = []
        if isinstance(parsed, dict):
            insights = parsed.get("insights") or []
            # Store supplemental sections inside metadata for UI consumption
            extras = {
                "channel_watchlist": parsed.get("channel_watchlist"),
                "channel_recommendations": parsed.get("channel_recommendations"),
            }
            if extras.get("channel_watchlist") or extras.get("channel_recommendations"):
                if "channel_summary" not in self.channel_summary:
                    self.channel_summary["ai_recommendations"] = extras
                else:
                    self.channel_summary.setdefault("ai_recommendations", {}).update(extras)

            if isinstance(insights, list):
                return insights

        if isinstance(parsed, list):
            return parsed

        return []

    def get_insights(self) -> List[Dict[str, Any]]:
        """Get current insights from the agent"""
        return self.channel_cache

    def get_channel_summary(self) -> Dict[str, Any]:
        """Return the last computed channel summary"""
        return self.channel_summary


