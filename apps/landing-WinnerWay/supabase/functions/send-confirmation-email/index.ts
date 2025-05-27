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
    const subject = "You're in! 🎾 Your first swing with Winner Way";

  const resendResponse = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from: "Winner Way <updates@winnerway.pro>",
      to: email,
      subject: subject,
      html: `
        <div style="font-family: Arial, sans-serif; background: #f4f4f4; padding: 20px;">
          <div style="max-width: 600px; margin: auto; background: white; padding: 30px; border-radius: 8px;">
            <img src="https://winnerway.pro/banner.png?v=2" alt="Winner Way Banner" style="max-width: 100%; height: auto; margin-bottom: 20px;" />
            <h1 style="color: #1a202c;">You're in! 🎾</h1>
            <p style="font-size: 16px; color: #333;">Hey there 👋</p>
            <p style="font-size: 16px; color: #333;">
              Thanks for trying out <strong>Winner Way</strong>. You’ve just taken the first step toward improving your game with AI.
            </p>
            <p style="font-size: 16px; color: #333;">
              This was just a quick preview of what’s coming. We're testing things fast, and you’re part of that journey. Stay tuned — we’ll let you know as soon as full access is ready.
            </p>
            <p style="font-size: 16px; color: #333;">
              Meanwhile, feel free to test again or send us a message if you'd like personalized feedback.
            </p>
            <p>
              <a href="https://www.instagram.com/winnerwayai/?hl=en" target="_blank" style="font-size: 16px; color: #4c51bf;">
                👉 Follow us on Instagram to see behind-the-scenes, tips and updates
              </a>
            </p>
            <p style="font-size: 16px; color: #333;">
              Let’s keep improving — together. 🏆
            </p>
            <p style="font-size: 16px; color: #333;">
              — <strong>Emilio Zavala</strong><br>
              Founder & CEO, Winner Way
            </p>
          </div>
        </div>
      `,
    }),
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