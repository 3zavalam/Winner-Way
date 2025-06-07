import os
import stripe
from flask import Blueprint, request, jsonify

# Asumiendo que tienes un cliente de Supabase configurado así:
try:
    from integrations.supabase import supabase
except ImportError:
    supabase = None

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")

stripe_bp = Blueprint("stripe_bp", __name__)

@stripe_bp.route("/api/create-checkout-session", methods=["POST"])
def create_checkout_session():
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            line_items=[{
                "price": os.environ.get("STRIPE_PRICE_ID"), # Carga el Price ID desde variables de entorno
                "quantity": 1,
            }],
            metadata={"user_id": user_id},
            # URLs basadas en tu configuración general
            success_url="https://www.winnerway.pro/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="https://www.winnerway.pro/cancelled",
        )
        return jsonify({"url": session.url})
    except Exception as e:
        return jsonify({"error": str(e)}), 400