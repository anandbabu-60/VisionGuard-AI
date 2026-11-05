"""
Email and External Notification Module for VisionGuard AI.
Uses Python's standard library (smtplib, email) to send security alert notifications
with embedded snapshot image attachments.
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
from datetime import datetime


class EmailNotifier:
    """
    Dispatches security email alerts when high-severity incidents occur.
    """

    def __init__(self):
        self.enabled = os.getenv("EMAIL_ALERTS_ENABLED", "False").lower() in ("true", "1", "yes")
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_pass = os.getenv("SMTP_PASS", "")
        self.recipient = os.getenv("ALERT_RECIPIENT_EMAIL", "")

    def send_alert_email(
        self,
        alert_type,
        description,
        camera_id="CAM_01",
        severity="HIGH",
        snapshot_path=None,
    ):
        """
        Sends an HTML-formatted security alert with attached snapshot.

        Returns:
            bool: True if sent successfully, False otherwise.
        """
        if not self.enabled:
            # Email alerts disabled in configuration
            return False

        if not self.smtp_user or not self.smtp_pass or not self.recipient:
            print("[WARNING] Email notifications enabled but SMTP credentials are not configured in .env.")
            return False

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"🚨 [VisionGuard AI Alert - {severity}] {alert_type} at {camera_id}"

        # Create MIME message
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = self.smtp_user
        msg["To"] = self.recipient

        # HTML Email Body
        html = f"""
        <html>
            <body style="font-family: Arial, sans-serif; background-color: #0b0f19; color: #ffffff; padding: 20px;">
                <div style="background-color: #121826; border: 1px solid #da3633; border-radius: 8px; padding: 20px; max-width: 600px;">
                    <h2 style="color: #ff3b30; margin-top: 0;">🛡️ VisionGuard AI Security Alert</h2>
                    <hr style="border: 0; border-top: 1px solid #30363d;">
                    <p><strong>Incident Type:</strong> <span style="color: #00ff88;">{alert_type}</span></p>
                    <p><strong>Severity:</strong> <span style="color: #ff3b30; font-weight: bold;">{severity}</span></p>
                    <p><strong>Camera Location:</strong> {camera_id}</p>
                    <p><strong>Timestamp:</strong> {now_str}</p>
                    <p><strong>Description:</strong> {description}</p>
                    <hr style="border: 0; border-top: 1px solid #30363d;">
                    <p style="font-size: 12px; color: #8b949e;">Automated notification from VisionGuard AI Surveillance System.</p>
                </div>
            </body>
        </html>
        """
        msg.attach(MIMEText(html, "html"))

        # Attach image snapshot if available
        if snapshot_path and Path(snapshot_path).exists():
            try:
                with open(snapshot_path, "rb") as img_f:
                    img_data = img_f.read()
                    image_mime = MIMEImage(img_data)
                    image_mime.add_header("Content-Disposition", "attachment", filename=Path(snapshot_path).name)
                    msg.attach(image_mime)
            except Exception as e:
                print(f"[WARNING] Could not attach snapshot to email: {e}")

        # Send via SMTP
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            print(f"[INFO] Alert email dispatched successfully to {self.recipient}.")
            return True
        except Exception as err:
            print(f"[ERROR] Failed to dispatch alert email: {err}")
            return False

