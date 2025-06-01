// apps/landing-WinnerWay/src/components/StartTrialPromo.tsx

import React, { useState, useEffect } from "react";
import { useSession } from "@/context/SessionContext";
import { supabase } from "@/lib/supabaseClient";
import { toast } from "sonner";

const StartTrialPromo: React.FC = () => {
  const { user, trialActive, loading } = useSession();
  const [trialStarted, setTrialStarted] = useState(false);
  const [loadingBtn, setLoadingBtn] = useState(false);

  // Mostrar el toast solo si el usuario está cargado, ya terminó de cargar el perfil,
  // y el trial no está activo
  useEffect(() => {
    if (!user) return;
    if (loading) return;
    if (!trialActive) {
      toast.warning("Your free trial has ended. Scroll down to upgrade.");
    }
  }, [user, loading, trialActive]);

  const handleStartTrial = async () => {
    if (!user) return;

    // Verificar si el usuario ya tiene un trial_end en la tabla "users"
    const { data: profile, error: profileError } = await supabase
      .from("users")
      .select("trial_end")
      .eq("id", user.id)
      .single();

    if (profileError) {
      toast.error("Error verificando el estado del trial.");
      return;
    }

    // Si ya existe trial_end (incluso si expiró), no permitimos iniciar otro
    if (profile?.trial_end) {
      return;
    }

    setLoadingBtn(true);

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

    setLoadingBtn(false);
  };

  const handleCheckout = async () => {
    if (!user) {
      toast.error("You must be logged in to purchase.");
      return;
    }

    const {
      data: { session },
    } = await supabase.auth.getSession();
    const accessToken = session?.access_token || "";

    const res = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/api/create-checkout-session`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ user_id: user.id }),
      }
    );

    const data = await res.json();
    if (data.url) {
      window.location.href = data.url;
    } else {
      toast.error("Failed to redirect to checkout.");
    }
  };

  if (loading) {
    return null;
  }

  return (
    <section
      id="start-trial"
      className="py-16 px-6 text-center bg-white/90 rounded-xl max-w-3xl mx-auto shadow-md border border-winner-green/10 mt-12"
    >
      <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-4">
        Unlock WinnerWay Pro
      </h2>

      <p className="text-winner-green/80 text-base mb-6">
        🎾 Get personalized AI feedback
        <br />
        💡 Tailored drills based on your stroke
        <br />
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

      {trialActive ? (
        <p className="text-green-600 font-medium">✅ Tu trial está activo</p>
      ) : trialStarted ? (
        <p className="text-green-600 font-medium">
          ✅ Your free trial is now active!
        </p>
      ) : (
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <button
            onClick={handleStartTrial}
            disabled={loadingBtn}
            className="btn-primary"
          >
            {loadingBtn ? "Activating..." : "Start Free Trial"}
          </button>
          <button onClick={handleCheckout} className="btn-secondary">
            Buy Now – $29/year
          </button>
        </div>
      )}

      <p className="text-xs text-winner-green/60 mt-4">
        No credit card required for trial
      </p>
    </section>
  );
};

export default StartTrialPromo;
