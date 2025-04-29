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
    console.log("📥 Body received in function:", body);
    const email = body?.email;
    console.log("✉️ Extracted email:", email);

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

    if (!RESEND_API_KEY) {
      return new Response(JSON.stringify({ error: "Missing RESEND_API_KEY" }), {
        status: 400,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Content-Type": "application/json",
        },
      });
    }

    const resendResponse = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${RESEND_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        from: "Winner Way <updates@winnerway.pro>",
        to: email,
        subject: "You’re in! 🥳 Welcome to Winner Way Beta 🎾",
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
                Your feedback as a beta tester will shape everything.
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

    return new Response(JSON.stringify({ message: "Email sent successfully!", body }), {
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