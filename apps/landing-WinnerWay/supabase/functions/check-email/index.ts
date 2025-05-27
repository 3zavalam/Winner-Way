// supabase/functions/check-email/index.ts
import { serve } from "https://deno.land/std@0.192.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

serve(async (req) => {
  if (req.method !== "POST") {
    return new Response("Method Not Allowed", { status: 405 });
  }

  const { email } = await req.json();
  if (!email || typeof email !== "string") {
    return new Response(JSON.stringify({ error: "Invalid email" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
  );

  const { data: lead, error: leadError } = await supabase
    .from("leads")
    .select("id")
    .eq("email", email)
    .maybeSingle();

  if (leadError) {
    return new Response(JSON.stringify({ error: leadError.message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Límite por día
  const DAILY_LIMIT = 3;
  const now = new Date();
  const startOfDay = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate())).toISOString();

  const { count, error: countError } = await supabase
    .from("usage_logs")
    .select("*", { count: "exact", head: true })
    .eq("email", email)
    .gte("called_at", startOfDay);

  if (countError) {
    return new Response(JSON.stringify({ error: countError.message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  const overLimit = (count ?? 0) >= DAILY_LIMIT;

  return new Response(
    JSON.stringify({
      exists: !!lead,
      calls_today: count,
      over_limit: overLimit
    }),
    { status: 200, headers: { "Content-Type": "application/json" } }
  );
});
