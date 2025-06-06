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

@stripe_bp.route('/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    endpoint_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")

    if not endpoint_secret:
        print("Stripe webhook secret is not configured.")
        return 'Configuration error', 500

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError:
        # Invalid payload
        return 'Invalid payload', 400
    except stripe.error.SignatureVerificationError:
        # Invalid signature
        return 'Invalid signature', 400

    # Handle the checkout.session.completed event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        user_id = session.get('metadata', {}).get('user_id')

        if user_id and supabase:
            try:
                # Actualizar el perfil del usuario para dar acceso ilimitado
                data, error = supabase.table('profiles').update({'has_unlimited_access': True}).eq('id', user_id).execute()
                if error:
                    print(f"Error updating Supabase for user {user_id}: {error}")
                else:
                    print(f"User {user_id} granted unlimited access.")
            except Exception as e:
                print(f"An exception occurred while updating user {user_id} in Supabase: {e}")
        elif not user_id:
            print("Webhook received for session without user_id in metadata.")
        else:
            print("Supabase client not available, cannot update user.")


    return 'Success', 200
