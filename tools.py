import os

from dotenv import load_dotenv

load_dotenv()


def send_whatsapp_message(message: str, to: str | None = None):
    """
    WhatsApp sending function.

    If Twilio credentials are unavailable, this acts as a mock
    and prints the message.
    """

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_WHATSAPP_FROM")

    to_number = to or os.getenv("CUSTOMER_WHATSAPP")

    # Development / mock mode
    if not all([
        account_sid,
        auth_token,
        from_number,
        to_number
    ]):
        print("\n========== WHATSAPP ==========")
        print(message)
        print("==============================\n")

        return {
            "status": "mock_sent",
            "message": message
        }

    from twilio.rest import Client

    client = Client(
        account_sid,
        auth_token
    )

    result = client.messages.create(
        from_=from_number,
        body=message,
        to=to_number
    )

    return {
        "status": "sent",
        "message_sid": result.sid
    }