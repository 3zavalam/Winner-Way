import { createContext, useContext, useEffect, useState } from "react";
import { supabase } from "@/lib/supabaseClient";

interface SessionContextType {
  user: any;
  trialActive: boolean;
  loading: boolean;
}

const SessionContext = createContext<SessionContextType>({
  user: null,
  trialActive: false,
  loading: true,
});

function generateReferralCode(email: string): string {
  const base = email.split("@")[0].replace(/[^a-zA-Z0-9]/g, "").slice(0, 8);
  return base + Math.floor(1000 + Math.random() * 9000).toString();
}

export const SessionProvider = ({ children }: { children: React.ReactNode }) => {
  const [user, setUser] = useState<any>(null);
  const [trialActive, setTrialActive] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const init = async () => {
      const { data } = await supabase.auth.getUser();
      const user = data?.user || null;
      setUser(user);

      if (user) {
        // Obtener datos actuales del usuario
        const { data: userInfo } = await supabase
          .from("users")
          .select("*")
          .eq("id", user.id)
          .single();

        // Verificar trial
        if (userInfo?.trial_end) {
          const now = new Date();
          const trialEnd = new Date(userInfo.trial_end);
          setTrialActive(trialEnd > now);
        }

        // Generar referral_code si no existe
        if (!userInfo?.referral_code) {
          const newCode = generateReferralCode(user.email);
          await supabase
            .from("users")
            .update({ referral_code: newCode })
            .eq("id", user.id);
        }

        // Aplicar Buddy Pass si hay
        const referralCode = localStorage.getItem("pendingReferralCode");

        if (referralCode && !userInfo?.referred_by) {
          const { data: referrer, error: referrerError } = await supabase
            .from("users")
            .select("id")
            .eq("referral_code", referralCode)
            .neq("id", user.id)
            .single();

          if (referrer && !referrerError) {
            const newTrialEnd = new Date();
            newTrialEnd.setDate(newTrialEnd.getDate() + 7);

            await supabase
              .from("users")
              .update({
                trial_end: newTrialEnd.toISOString(),
                referred_by: referrer.id,
              })
              .eq("id", user.id);

            console.log("✅ Buddy Pass applied");
            localStorage.removeItem("pendingReferralCode");
            setTrialActive(true);
          } else {
            console.log("⚠️ Invalid or self-referral code");
            localStorage.removeItem("pendingReferralCode");
          }
        }
      }

      setLoading(false);
    };

    init();

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user || null);
    });

    return () => {
      listener.subscription.unsubscribe();
    };
  }, []);

  return (
    <SessionContext.Provider value={{ user, trialActive, loading }}>
      {children}
    </SessionContext.Provider>
  );
};

export const useSession = () => useContext(SessionContext);