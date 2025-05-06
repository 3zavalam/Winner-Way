import { serve } from "https://deno.land/std@0.192.0/http/server.ts";

serve(async (req) => {
  // ✅ Manejo de CORS preflight
  if (req.method === "OPTIONS") {
    return new Response("OK", {
      status: 200,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
      },
    });
  }

  try {
    if (req.headers.get("content-type") !== "application/json") {
      return new Response(JSON.stringify({ error: "Invalid content-type" }), {
        status: 400,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Content-Type": "application/json",
        },
      });
    }

    const body = await req.json();
    const email = body?.email;

    if (!email || typeof email !== "string" || !email.includes("@")) {
      return new Response(JSON.stringify({ error: "Invalid or missing email", body }), {
        status: 400,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Content-Type": "application/json",
        },
      });
    }

    const RESEND_API_KEY = Deno.env.get("RESEND_API_KEY");
    const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");

    if (!RESEND_API_KEY || !SUPABASE_SERVICE_ROLE_KEY) {
      return new Response(JSON.stringify({ error: "Missing required secrets" }), {
        status: 400,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Content-Type": "application/json",
        },
      });
    }

    // ✅ Enviar el correo
    const subject = "You’re in! 🥳 Welcome to Winner Way Beta 🎾"
    const resendResponse = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${RESEND_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        from: "Winner Way <updates@winnerway.pro>",
        to: email,
        subject: "You're in! 🎾",
        html: `
          <div style="font-family: Arial, sans-serif; background: #f4f4f4; padding: 20px;">
            <div style="max-width: 600px; margin: auto; background: white; padding: 30px; border-radius: 8px;">
              <img src="https://winnerway.pro/banner.png" alt="Winner Way Logo" style="max-width: 100%; height: auto; margin-bottom: 20px;" />
              <h1 style="color: #1a202c;">You're in! 🎾</h1>
              <p style="font-size: 16px; color: #333;">Hey,</p>
              <p style="font-size: 16px; color: #333;">
                I’m <strong>Emilio Zavala</strong>, founder of <strong>Winner Way</strong> — and I just wanted to personally say thanks for signing up.
              </p>
              <p style="font-size: 16px; color: #333;">
                You're now part of a small group getting early access to what we're building: a new way to train smarter, stay motivated, and win more — both on and off the court.
              </p>
              <p style="font-size: 16px; color: #333;">
                We’re not just launching an app. We’re creating a mindset.  
                And your feedback will help us shape it.
              </p>
              <p style="font-size: 16px; color: #333;">
                Want a free report on your game? Just reply to this email with a short video of you playing and the shot you’d like us to check — and we’ll send you personalized feedback.
              </p>
              <p>
                <a href="https://www.instagram.com/winnerwayai/?hl=en" target="_blank" style="font-size: 16px; color: #4c51bf;">
                  👉 Follow Winner Way on Instagram
                </a>
              </p>
              <p style="font-size: 16px; color: #333;">
                Thanks for believing in what we’re building.<br><br>
                Let’s win, together. 🏆
              </p>
              <p style="font-size: 16px; color: #333;">
                — <strong>Emilio Zavala</strong><br>
                Founder & CEO, Winner Way
              </p>
            </div>
          </div>
        `,
      })      
    });

    const data = await resendResponse.json();

    if (!resendResponse.ok) {
      console.error("❌ Failed to send email via Resend:", data);
      return new Response(JSON.stringify({ error: data }), {
        status: resendResponse.status,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Content-Type": "application/json",
        },
      });
    }

    // ✅ Registrar en tabla email_logs
    const insertLogResponse = await fetch("https://gxpmjqbxtlgkzemdyfwl.supabase.co/rest/v1/email_logs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
        "Prefer": "return=representation"
      },
      body: JSON.stringify({
        email: email,
        sent_at: new Date().toISOString(),
        subject: subject
      }),
    });

    if (!insertLogResponse.ok) {
      console.error("❌ Error al insertar en email_logs:", await insertLogResponse.text());
    }

    return new Response(JSON.stringify({ message: "Email sent successfully!" }), {
      status: 200,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json",
      },
    });

  } catch (error) {
    console.error("🔥 Function error:", error);
    return new Response(JSON.stringify({ error: "Internal server error", details: String(error) }), {
      status: 500,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json",
      },
    });
  }
});