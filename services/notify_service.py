"""Notification backends. Credentials live in app.storage.general; backends
are opt-in via an 'enabled' flag so a missing/untested config is a no-op.

Currently: Gmail SMTP only. To send to multiple recipients, put a
comma-separated list in the 'send to' field — SMTP delivers to all of them
in a single message."""
import smtplib
from email.mime.text import MIMEText

from nicegui import app


def _cfg():
    return app.storage.general


def _split_recipients(raw):
    return [r.strip() for r in (raw or '').split(',') if r.strip()]


def send_gmail(subject, body):
    cfg = _cfg()
    if not cfg.get('notify_gmail_enabled'):
        return False, 'disabled'
    user = (cfg.get('notify_gmail_user') or '').strip()
    pw = cfg.get('notify_gmail_password') or ''
    to_raw = (cfg.get('notify_gmail_to') or user).strip()
    recipients = _split_recipients(to_raw) or [user]
    if not user or not pw or not recipients:
        return False, 'missing credentials'
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = user
    msg['To'] = ', '.join(recipients)
    try:
        with smtplib.SMTP('smtp.gmail.com', 587, timeout=20) as s:
            s.starttls()
            s.login(user, pw)
            s.sendmail(user, recipients, msg.as_string())
        return True, None
    except Exception as e:
        return False, str(e)[:300]


def send(subject, body):
    """Fan-out to every enabled backend. Returns {backend: (ok, err)}."""
    return {
        'gmail': send_gmail(subject, body),
    }


def send_test(backend):
    test_body = 'HPML Queue test notification. If you see this, the backend is wired correctly.'
    if backend == 'gmail':
        return send_gmail('HPML Queue Test', test_body)
    return False, f'unknown backend: {backend}'


def enabled_backends():
    cfg = _cfg()
    out = []
    if cfg.get('notify_gmail_enabled'):
        out.append('gmail')
    return out
