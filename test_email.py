import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

# Load .env file
load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM = os.getenv("SMTP_FROM")
EMAIL_TO = os.getenv("EMAIL_TO")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

# Validate required config
missing = [
    name for name, value in {
        "SMTP_HOST": SMTP_HOST,
        "SMTP_PORT": SMTP_PORT,
        "SMTP_USER": SMTP_USER,
        "SMTP_PASSWORD": SMTP_PASSWORD,
        "SMTP_FROM": SMTP_FROM,
        "EMAIL_TO": EMAIL_TO,
    }.items()
    if not value
]

if missing:
    raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

# Build email
msg = EmailMessage()
msg["Subject"] = "SMTP Test"
msg["From"] = SMTP_FROM
msg["To"] = EMAIL_TO
msg.set_content("SMTP is working via .env configuration")

# Send email
server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
try:
    if SMTP_USE_TLS:
        server.starttls()
    server.login(SMTP_USER, SMTP_PASSWORD)
    server.send_message(msg)
finally:
    server.quit()

print("Email sent successfully ✔")
