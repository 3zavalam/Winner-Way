import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

const Success = () => {
  const [searchParams] = useSearchParams();
  const sessionId = searchParams.get("session_id");
  const [message, setMessage] = useState("Verifying your payment...");

  useEffect(() => {
    if (!sessionId) return;

    const verifyPayment = async () => {
      try {
        const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/api/verify-session`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId }),
        });

        const data = await res.json();
        if (data.success) {
          setMessage("✅ Payment verified. Welcome to WinnerWay Pro!");

          // ⏱ Redirige automáticamente al home después de 3 segundos
          setTimeout(() => {
            window.location.href = "/";
          }, 3000);
        } else {
          setMessage("❌ Payment could not be verified.");
        }
      } catch (err) {
        setMessage("⚠️ Error verifying payment.");
      }
    };

    verifyPayment();
  }, [sessionId]);

  return (
    <section className="p-8 text-center bg-winner-cream min-h-screen flex flex-col items-center justify-center">
      <h1 className="text-2xl font-bold text-winner-green mb-4">Thank you!</h1>
      <p className="text-winner-green text-lg">{message}</p>
      <p className="text-sm text-winner-green/70 mt-4">
        Redirecting to the app...
      </p>
    </section>
  );
};

export default Success;
