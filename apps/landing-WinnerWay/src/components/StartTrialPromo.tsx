import { useSession } from "@/context/SessionContext";
import { supabase } from "@/lib/supabaseClient";
import { useState } from "react";

const StartTrialPromo = () => {
  const { user } = useSession();
  const [trialStarted, setTrialStarted] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleStartTrial = async () => {
    if (!user?.id) return;

    setLoading(true);
    const newTrialEnd = new Date();
    newTrialEnd.setDate(newTrialEnd.getDate() + 7);

    const { error } = await supabase
      .from("users")
      .update({ trial_end: newTrialEnd.toISOString() })
      .eq("id", user.id);

    if (!error) {
      setTrialStarted(true);
      setTimeout(() => {
        window.location.href = "/";
      }, 2000); // espera 2 segundos
    }

    setLoading(false);
  };

  const handleCheckout = async () => {
    const res = await fetch("http://localhost:5050/api/create-checkout-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: user?.id || "unknown" }),
    });

    const data = await res.json();
    if (data.url) {
      window.location.href = data.url;
    } else {
      alert("Failed to redirect to checkout.");
    }
  };

  return (
    <section className="py-16 px-6 text-center bg-white/90 rounded-xl max-w-3xl mx-auto shadow-md border border-winner-green/10 mt-12">
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

      {trialStarted ? (
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