"""Notification backends. Credentials live in app.storage.general; all backends
are opt-in via an 'enabled' flag so a missing/untested config is a no-op."""
import smtplib
from email.mime.text import MIMEText

import requests
from nicegui import app


def _cfg():
    return app.storage.general


def _masked(val, keep=4):
    if not val:
        return ''
    if len(val) <= keep:
        return '*' * len(val)
    return '*' * (len(val) - keep) + val[-keep:]


def send_gmail(subject, body):
    cfg = _cfg()
    if not cfg.get('notify_gmail_enabled'):
        return False, 'disabled'
    user = (cfg.get('notify_gmail_user') or '').strip()
    pw = cfg.get('notify_gmail_password') or ''
    to = (cfg.get('notify_gmail_to') or user).strip()
    if not user or not pw or not to:
        return False, 'missing credentials'
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = user
    msg['To'] = to
    try:
        with smtplib.SMTP('smtp.gmail.com', 587, timeout=20) as s:
            s.starttls()
            s.login(user, pw)
            s.send_message(msg)
        return True, None
    except Exception as e:
        return False, str(e)[:300]


def send_whatsapp(body):
    cfg = _cfg()
    if not cfg.get('notify_whatsapp_enabled'):
        return False, 'disabled'
    sid = (cfg.get('notify_twilio_sid') or '').strip()
    token = cfg.get('notify_twilio_token') or ''
    from_num = (cfg.get('notify_twilio_from') or '').strip()
    to_num = (cfg.get('notify_twilio_to') or '').strip()
    if not (sid and token and from_num and to_num):
        return False, 'missing credentials'
    # Twilio expects `whatsapp:+NNN` prefix for WhatsApp endpoints.
    if not from_num.startswith('whatsapp:'):
        from_num = f'whatsapp:{from_num}'
    if not to_num.startswith('whatsapp:'):
        to_num = f'whatsapp:{to_num}'
    url = f'https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json'
    try:
        r = requests.post(
            url,
            auth=(sid, token),
            data={'From': from_num, 'To': to_num, 'Body': body},
            timeout=20,
        )
        if r.status_code in (200, 201):
            return True, None
        return False, f'twilio status {r.status_code}: {r.text[:240]}'
    except Exception as e:
        return False, str(e)[:300]


def send(subject, body):
    """Fan-out to every enabled backend. Returns {backend: (ok, err)}.
    WhatsApp is body-only (no subject); Gmail uses both."""
    return {
        'gmail': send_gmail(subject, body),
        'whatsapp': send_whatsapp(body),
    }


def send_test(backend):
    test_body = 'HPML Queue test notification. If you see this, the backend is wired correctly.'
    if backend == 'gmail':
        return send_gmail('HPML Queue Test', test_body)
    if backend == 'whatsapp':
        return send_whatsapp(test_body)
    return False, f'unknown backend: {backend}'


def enabled_backends():
    cfg = _cfg()
    out = []
    if cfg.get('notify_gmail_enabled'):
        out.append('gmail')
    if cfg.get('notify_whatsapp_enabled'):
        out.append('whatsapp')
    return out
