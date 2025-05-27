import { useState } from "react";
import { supabase } from "@/lib/supabaseClient";

export default function LoginFormWithBuddyPass() {
  const [email, setEmail] = useState("");
  const [buddyPass, setBuddyPass] = useState("");
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = async () => {
    setLoading(true);
    setError("");

    if (!email || !email.includes("@")) {
      setError("Please enter a valid email.");
      setLoading(false);
      return;
    }

    try {
      // Guardar buddy pass en localStorage (si existe)
      if (buddyPass.trim()) {
        localStorage.setItem("pendingReferralCode", buddyPass.trim());
      }

      console.log("Attempting to sign in with OTP for email:", email);
      
      // Primero verificar si el usuario ya existe
      const { data: existingUser, error: checkError } = await supabase
        .from('users')
        .select('id')
        .eq('email', email)
        .single();

      console.log("Existing user check:", { existingUser, checkError });

      const { error: signInError, data } = await supabase.auth.signInWithOtp({
        email,
        options: {
          emailRedirectTo: window.location.origin,
          data: {
            email: email
          }
        }
      });

      console.log("SIGN IN RESULT", { error: signInError, data });

      if (signInError) {
        console.error("Sign in error details:", signInError);
        if (signInError.message.includes("Database error")) {
          setError("There was a problem creating your account. Please try again or contact support.");
        } else {
          setError("Supabase error: " + signInError.message);
        }
      } else {
        setSent(true);
      }
    } catch (err) {
      console.error("Catch error:", err);
      setError("Something went wrong. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white/90 rounded-xl shadow-lg p-6 md:p-8 border border-winner-green/10 max-w-md mx-auto text-center">
      <h2 className="text-xl font-bold text-winner-green mb-4">Get Started</h2>
      {!sent ? (
        <>
          <div className="space-y-4">
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="your@email.com"
              className="w-full p-3 border rounded text-sm"
            />
            <input
              type="text"
              value={buddyPass}
              onChange={(e) => setBuddyPass(e.target.value)}
              placeholder="Got a Buddy Pass? (optional)"
              className="w-full p-3 border rounded text-sm"
            />
            <button
              onClick={handleLogin}
              disabled={loading}
              className="btn-primary w-full"
            >
              {loading ? "Sending..." : "Send Magic Link"}
            </button>
            {error && <p className="text-red-500 text-sm">{error}</p>}
          </div>
        </>
      ) : (
        <p className="text-winner-green/80 text-sm">
          ✅ Magic link sent! Check your inbox to continue.
        </p>
      )}
    </div>
  );
}
