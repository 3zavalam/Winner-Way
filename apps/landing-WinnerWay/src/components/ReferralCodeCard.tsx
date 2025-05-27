import { useEffect, useState } from "react";
import { supabase } from "../lib/supabaseClient";

const ReferralCodeCard = ({ userId }: { userId: string }) => {
  const [referralCode, setReferralCode] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const fetchOrCreateCode = async () => {
      const { data, error } = await supabase
        .from("users")
        .select("referral_code")
        .eq("id", userId)
        .single();

      if (data?.referral_code) {
        setReferralCode(data.referral_code);
      } else {
        // generar código
        const namePart = userId.slice(0, 3).toUpperCase();
        const rand = Math.floor(100 + Math.random() * 900);
        const newCode = `${namePart}${rand}`;

        const { error: updateError } = await supabase
          .from("users")
          .update({ referral_code: newCode })
          .eq("id", userId);

        if (!updateError) {
          setReferralCode(newCode);
        }
      }
    };

    fetchOrCreateCode();
  }, [userId]);

  const handleCopy = () => {
    if (referralCode) {
      navigator.clipboard.writeText(`https://winnerway.app?ref=${referralCode}`);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (!referralCode) return null;

  return (
    <div className="bg-white/90 border border-winner-green/20 rounded-xl p-6 text-center shadow-sm max-w-md mx-auto">
      <h3 className="text-lg font-bold text-winner-green mb-2">
        🎁 Your Buddy Pass
      </h3>
      <p className="text-winner-green/80 text-sm mb-3">
        Share this code with friends to give them +7 days trial!
      </p>
      <div className="bg-winner-green/5 text-winner-green font-mono text-sm px-4 py-2 rounded mb-4">
        {referralCode}
      </div>
    </div>
  );
};

export default ReferralCodeCard;
