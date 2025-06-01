// apps/landing-WinnerWay/src/context/SessionContext.tsx

import { createContext, useContext, useEffect, useState } from "react";
import { supabase } from "@/lib/supabaseClient";

// ─── Definición del tipo del contexto ─────────────────────────────────────────
interface SessionContextType {
  /** El usuario que provee Supabase Auth (o null si no está logueado) */
  user: any;
  /** true si profile.trial_end existe y está en el futuro */
  trialActive: boolean;
  /** true mientras se está obteniendo la sesión o el perfil */
  loading: boolean;
}

const SessionContext = createContext<SessionContextType>({
  user: null,
  trialActive: false,
  loading: true,
});

// ─── Función para generar un referral_code (lo dejas igual) ────────────────────
function generateReferralCode(email: string): string {
  const base = email.split("@")[0].replace(/[^a-zA-Z0-9]/g, "").slice(0, 8);
  return base + Math.floor(1000 + Math.random() * 9000).toString();
}

// ─── Provider ───────────────────────────────────────────────────────────────────
export const SessionProvider = ({ children }: { children: React.ReactNode }) => {
  // 1) “user” guarda el objeto de Supabase Auth (session.user)
  const [user, setUser] = useState<any>(null);
  // 2) “trialActive” se calculará según la fecha “trial_end” en tu tabla “users”
  const [trialActive, setTrialActive] = useState(false);
  // 3) “loading” es true mientras obtenemos Auth y luego el perfil de la tabla
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    /**
     * Función principal que:
     * 1) obtiene la sesión actual de Supabase Auth (supabase.auth.getUser())
     * 2) si existe usuario, lee su fila en la tabla “users” para ver “trial_end”
     * 3) genera referral_code si no existe, y aplica Buddy Pass si hay código pendiente
     */
    const init = async () => {
      // ─── 1. Obtener datos de Supabase Auth ────────────────────────────────
      const { data } = await supabase.auth.getUser();
      const sessionUser = data?.user || null;
      setUser(sessionUser);

      if (sessionUser) {
        // ─── 2. Traer la fila completa desde la tabla “users” ─────────────────
        const { data: userInfo, error: userInfoError } = await supabase
          .from("users")
          .select("*")
          .eq("id", sessionUser.id)
          .single();

        if (userInfoError) {
          console.error("Error al leer perfil de usuario:", userInfoError.message);
        } else if (userInfo) {
          // 2.1 Verificar trial: si existe userInfo.trial_end, compararlo con “hoy”
          if (userInfo.trial_end) {
            const now = new Date();
            const trialEndDate = new Date(userInfo.trial_end);
            setTrialActive(trialEndDate > now);
          }

          // 2.2 Generar referral_code si aún no existe en la tabla “users”
          if (!userInfo.referral_code) {
            const newCode = generateReferralCode(sessionUser.email);
            await supabase
              .from("users")
              .update({ referral_code: newCode })
              .eq("id", sessionUser.id);
            // (no guardamos el nuevo code en userInfo porque el contexto solo expone trialActive)
          }

          // 2.3 Aplicar Buddy Pass si hay "pendingReferralCode" en localStorage y no tiene “referred_by”
          const pendingCode = localStorage.getItem("pendingReferralCode");
          if (pendingCode && !userInfo.referred_by) {
            const { data: referrer, error: referrerError } = await supabase
              .from("users")
              .select("id")
              .eq("referral_code", pendingCode)
              .neq("id", sessionUser.id)
              .single();

            if (!referrerError && referrer) {
              // Dar 7 días adicionales de trial al usuario actual
              const newTrialEnd = new Date();
              newTrialEnd.setDate(newTrialEnd.getDate() + 7);

              await supabase
                .from("users")
                .update({
                  trial_end: newTrialEnd.toISOString(),
                  referred_by: referrer.id,
                })
                .eq("id", sessionUser.id);

              console.log("✅ Buddy Pass applied");
              localStorage.removeItem("pendingReferralCode");
              // Como ya extendimos trial, marcamos trialActive = true
              setTrialActive(true);
            } else {
              console.log("⚠️ Invalid or self-referral code");
              localStorage.removeItem("pendingReferralCode");
            }
          }
        }
      }

      // ─── 3. Finalmente, marcamos que ya terminamos de cargar ──────────────
      setLoading(false);
    };

    init();

    // ─── Listener para cambios en la sesión (login / logout) ────────────────
    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user || null);
      // Cuando detecte logout, user quedará en null y trialActive permanece false
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