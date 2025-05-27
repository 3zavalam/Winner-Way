import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/lib/supabaseClient";

export default function AuthCallback() {
  const navigate = useNavigate();

  useEffect(() => {
    const checkSession = async () => {
      const { data } = await supabase.auth.getSession();
      console.log("🔑 Session from callback:", data);

      if (data?.session) {
        navigate("/"); // redirige al home o donde prefieras
      } else {
        navigate("/login");
      }
    };

    checkSession();
  }, []);

  return (
    <div className="text-center mt-20 text-winner-green">
      🔄 Logging you in...
    </div>
  );
}
