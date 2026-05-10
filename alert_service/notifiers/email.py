"""Email notifier with batched digest support."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import aiosmtplib
import structlog

from shared.schemas import AlertEvent, AlertSeverity

logger = structlog.get_logger(__name__)


class EmailNotifier:
    """Send batched alert digest emails via SMTP.

    Collects alerts over a configurable interval (default 5 minutes)
    and sends a single HTML digest email to all configured recipients.
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
        use_tls: bool,
        from_address: str,
        recipients: list[str],
        batch_interval_seconds: int = 300,
    ) -> None:
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._smtp_user = smtp_user
        self._smtp_password = smtp_password
        self._use_tls = use_tls
        self._from_address = from_address
        self._recipients = recipients
        self._batch_interval = batch_interval_seconds
        self._pending_alerts: list[AlertEvent] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background flush task."""
        self._flush_task = asyncio.create_task(self._periodic_flush())
        await logger.ainfo(
            "email_notifier_started",
            smtp_host=self._smtp_host,
            recipients=self._recipients,
            batch_interval=self._batch_interval,
        )

    async def stop(self) -> None:
        """Flush remaining alerts and stop the background task."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Flush any remaining alerts
        await self._flush()
        await logger.ainfo("email_notifier_stopped")

    async def enqueue(self, alert: AlertEvent) -> None:
        """Add an alert to the pending batch.

        Args:
            alert: The alert event to include in the next digest.
        """
        async with self._lock:
            self._pending_alerts.append(alert)
            await logger.ainfo(
                "alert_enqueued_for_email",
                alert_id=str(alert.alert_id),
                pending_count=len(self._pending_alerts),
            )

    async def _periodic_flush(self) -> None:
        """Periodically flush the alert batch as a digest email."""
        while True:
            try:
                await asyncio.sleep(self._batch_interval)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception:
                await logger.aexception("email_flush_error")

    async def _flush(self) -> None:
        """Send all pending alerts as a digest email."""
        async with self._lock:
            if not self._pending_alerts:
                return
            alerts = list(self._pending_alerts)
            self._pending_alerts.clear()

        if not self._recipients:
            await logger.awarning("email_no_recipients_configured", alert_count=len(alerts))
            return

        html = self._build_html_digest(alerts)
        subject = self._build_subject(alerts)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._from_address
        msg["To"] = ", ".join(self._recipients)
        msg.attach(MIMEText(html, "html"))

        try:
            if self._use_tls:
                smtp = aiosmtplib.SMTP(
                    hostname=self._smtp_host,
                    port=self._smtp_port,
                    use_tls=True,
                )
            else:
                smtp = aiosmtplib.SMTP(
                    hostname=self._smtp_host,
                    port=self._smtp_port,
                )

            await smtp.connect()
            if self._smtp_user and self._smtp_password:
                await smtp.login(self._smtp_user, self._smtp_password)
            await smtp.send_message(msg)
            await smtp.quit()

            await logger.ainfo(
                "email_digest_sent",
                alert_count=len(alerts),
                recipients=self._recipients,
            )
        except Exception:
            await logger.aexception(
                "email_send_failed",
                alert_count=len(alerts),
            )

    @staticmethod
    def _build_subject(alerts: list[AlertEvent]) -> str:
        """Build the email subject line."""
        critical_count = sum(
            1 for a in alerts
            if a.severity in (AlertSeverity.CRITICAL, AlertSeverity.HIGH)
        )
        if critical_count > 0:
            return (
                f"[URGENT] Fraud Alert Digest: {len(alerts)} alerts "
                f"({critical_count} critical/high)"
            )
        return f"Fraud Alert Digest: {len(alerts)} alerts"

    @staticmethod
    def _severity_color(severity: AlertSeverity) -> str:
        """Return a CSS color for a severity level."""
        return {
            AlertSeverity.CRITICAL: "#dc3545",
            AlertSeverity.HIGH: "#fd7e14",
            AlertSeverity.MEDIUM: "#ffc107",
            AlertSeverity.LOW: "#17a2b8",
        }.get(severity, "#6c757d")

    @classmethod
    def _build_html_digest(cls, alerts: list[AlertEvent]) -> str:
        """Build an HTML email body with all alerts in a table.

        Args:
            alerts: List of alerts to include.

        Returns:
            HTML string for the email body.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        rows = []
        for alert in alerts:
            color = cls._severity_color(alert.severity)
            rows.append(f"""
            <tr>
                <td style="padding:8px;border:1px solid #dee2e6;">
                    <code>{str(alert.transaction_id)[:8]}...</code>
                </td>
                <td style="padding:8px;border:1px solid #dee2e6;">
                    <span style="background:{color};color:#fff;padding:2px 8px;
                    border-radius:4px;font-size:12px;font-weight:bold;">
                        {alert.severity.value.upper()}
                    </span>
                </td>
                <td style="padding:8px;border:1px solid #dee2e6;">
                    {alert.fraud_score:.4f}
                </td>
                <td style="padding:8px;border:1px solid #dee2e6;">
                    {alert.amount} {alert.currency}
                </td>
                <td style="padding:8px;border:1px solid #dee2e6;">
                    <code>{str(alert.user_id)[:8]}...</code>
                </td>
                <td style="padding:8px;border:1px solid #dee2e6;">
                    {alert.reason}
                </td>
                <td style="padding:8px;border:1px solid #dee2e6;">
                    {alert.created_at.strftime("%H:%M:%S")}
                </td>
            </tr>""")

        table_rows = "\n".join(rows)

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                        Roboto, sans-serif; margin: 0; padding: 20px;
                        background: #f8f9fa; }}
                .container {{ max-width: 900px; margin: 0 auto;
                              background: #fff; border-radius: 8px;
                              box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                              padding: 24px; }}
                h1 {{ color: #343a40; font-size: 22px; margin-top: 0; }}
                .summary {{ background: #f1f3f5; padding: 12px 16px;
                            border-radius: 6px; margin-bottom: 20px;
                            font-size: 14px; color: #495057; }}
                table {{ width: 100%; border-collapse: collapse;
                         font-size: 13px; }}
                th {{ background: #343a40; color: #fff; padding: 10px 8px;
                      text-align: left; font-size: 12px;
                      text-transform: uppercase; letter-spacing: 0.5px; }}
                tr:nth-child(even) {{ background: #f8f9fa; }}
                .footer {{ margin-top: 20px; font-size: 12px; color: #868e96;
                           text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Fraud Alert Digest</h1>
                <div class="summary">
                    <strong>{len(alerts)}</strong> alert(s) generated as of
                    <strong>{now}</strong>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Transaction</th>
                            <th>Severity</th>
                            <th>Score</th>
                            <th>Amount</th>
                            <th>User</th>
                            <th>Reason</th>
                            <th>Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
                <div class="footer">
                    Fraud Detection Platform &mdash; Alert Service
                </div>
            </div>
        </body>
        </html>
        """
