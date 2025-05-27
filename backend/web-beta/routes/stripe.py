import os
import stripe
from flask import Blueprint, request, jsonify

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")

stripe_bp = Blueprint("stripe_bp", __name__)

@stripe_bp.route("/api/create-checkout-session", methods=["POST"])
def create_checkout_session():
    data = request.get_json()
    user_id = data.get("user_id")

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            line_items=[{
                "price": "price_1RT1UmR1jlXbTT7lTYdPV1VQ",  # 👈 Reemplaza con tu price_id real
                "quantity": 1,
            }],
            metadata={"user_id": user_id},
            success_url="http://localhost:8080/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="https://tuapp.com/cancelled",
        )
        return jsonify({"url": session.url})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
