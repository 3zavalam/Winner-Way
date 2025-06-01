import { useSession } from "@/context/SessionContext";
import { supabase } from "@/lib/supabaseClient";
import { useState, useEffect } from "react";
import { toast } from "sonner"; // Asegúrate de tenerlo instalado y configurado

const StartTrialPromo = () => {
  const { user } = useSession();
  const [trialStarted, setTrialStarted] = useState(false);
  const [loading, setLoading] = useState(false);

  const now = new Date();
  const trialEnd = user?.trial_end ? new Date(user.trial_end) : null;

  const isTrialActive = trialEnd && trialEnd > now;
  const hasUsedTrial = !!trialEnd;
  const isSubscribed = user?.is_subscribed; // si lo manejas

  // ✅ Mostrar el toast SOLO si ya usó el trial y este ya terminó
  useEffect(() => {
    if (!user || !trialEnd) return;

    const trialExpired = new Date(trialEnd) <= new Date();
    if (hasUsedTrial && trialExpired) {
      toast.warning("Your free trial has ended. Scroll down to upgrade.");
    }
  }, [user]);

  const handleStartTrial = async () => {
    if (!user?.id || hasUsedTrial) return;

    setLoading(true);

    const newTrialEnd = new Date();
    newTrialEnd.setDate(newTrialEnd.getDate() + 7);

    const { error } = await supabase
      .from("users")
      .update({ trial_end: newTrialEnd.toISOString() })
      .eq("id", user.id);

    if (!error) {
      await supabase.auth.refreshSession();
      setTrialStarted(true);
      toast.success("Free trial activated! 🎉 Redirecting...");
      setTimeout(() => {
        window.location.href = "/";
      }, 2000);
    } else {
      toast.error("There was an error starting your trial.");
    }

    setLoading(false);
  };

  const handleCheckout = async () => {
    const res = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/api/create-checkout-session`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${user?.access_token || ""}`, // opcional si tu backend lo usa
        },
        body: JSON.stringify({ user_id: user?.id || "unknown" }),
      }
    );

    const data = await res.json();
    if (data.url) {
      window.location.href = data.url;
    } else {
      alert("Failed to redirect to checkout.");
    }
  };

  return (
    <section id="start-trial" className="py-16 px-6 text-center bg-white/90 rounded-xl max-w-3xl mx-auto shadow-md border border-winner-green/10 mt-12">
      <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-4">
        Unlock WinnerWay Pro
      </h2>

      <p className="text-winner-green/80 text-base mb-6">
        🎾 Get personalized AI feedback<br />
        💡 Tailored drills based on your stroke<br />
        🧑‍🎾 Compare yourself with 3 pro players
      </p>

      <div className="text-lg mb-4">
        <span className="line-through text-winner-green/50 mr-2">$99</span>
        <span className="text-winner-green font-bold">$29/year</span>
        <span className="text-sm ml-2 text-winner-green/60">(Founder Deal)</span>
      </div>

      <p className="text-sm text-winner-green/70 mb-6">
        7-day free trial – cancel anytime
      </p>

      {isTrialActive ? (
        <p className="text-green-600 font-medium">✅ Tu trial está activo</p>
      ) : hasUsedTrial ? (
        <p className="text-red-600 font-medium">⛔ Ya usaste tu trial gratuito</p>
      ) : trialStarted ? (
        <p className="text-green-600 font-medium">✅ Your free trial is now active!</p>
      ) : (
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <button
            onClick={handleStartTrial}
            disabled={loading}
            className="btn-primary"
          >
            {loading ? "Activating..." : "Start Free Trial"}
          </button>
          <button
            onClick={handleCheckout}
            className="btn-secondary"
          >
            Buy Now – $29/year
          </button>
        </div>
      )}

      <p className="text-xs text-winner-green/60 mt-4">No credit card required for trial</p>
    </section>
  );
};

export default StartTrialPromo;