import os
import stripe
from flask import Blueprint, request, jsonify
from integrations.supabase import supabase

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
webhook_bp = Blueprint("webhook_bp", __name__)

@webhook_bp.route("/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get("Stripe-Signature")
    secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    if not secret:
        return "Missing webhook secret", 500

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, secret)
    except ValueError:
        return "Invalid payload", 400
    except stripe.error.SignatureVerificationError:
        return "Invalid signature", 400

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session.get("metadata", {}).get("user_id")
        customer_id = session.get("customer")

        if user_id:
            try:
                supabase.table("users").update({
                    "sub_active": True,
                    "stripe_customer_id": customer_id
                }).eq("id", user_id).execute()
                print(f"✅ Sub activa para: {user_id}")
            except Exception as e:
                print(f"❌ Error Supabase: {e}")
        else:
            print("⚠️ No user_id en metadata.")

    return "Success", 200
