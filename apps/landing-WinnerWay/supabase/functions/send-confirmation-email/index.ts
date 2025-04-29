// archivo: functions/send-confirmation-email/index.ts

import { serve } from "https://deno.land/std@0.192.0/http/server.ts";

serve(async (req) => {
  if (req.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  const { email } = await req.json();

  if (!email || !email.includes('@')) {
    return new Response(JSON.stringify({ error: "Invalid email" }), {
      status: 400,
    });
  }

  const RESEND_API_KEY = Deno.env.get("RESEND_API_KEY");
  if (!RESEND_API_KEY) {
    return new Response(JSON.stringify({ error: "Missing RESEND_API_KEY" }), {
      status: 500,
    });
  }

  const resendResponse = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from: "Winner Way <onboarding@resend.dev>", 
      to: email,
      subject: "🏆 Welcome to Winner Way!",
      html: `
        <h1>Welcome to Winner Way!</h1>
        <p>Thanks for signing up for early access. Get ready to train smarter and win more matches. 🏆🎾</p>
      `,
    }),
  });

  if (!resendResponse.ok) {
    const errorData = await resendResponse.json();
    console.error("Resend error:", errorData);
    return new Response(JSON.stringify({ error: "Failed to send email" }), {
      status: 500,
    });
  }

  return new Response(JSON.stringify({ success: true }), {
    status: 200,
  });
});

// re_FQiNs1hP_7V8qpFaHs4qMCdPdGaF4N9Ew