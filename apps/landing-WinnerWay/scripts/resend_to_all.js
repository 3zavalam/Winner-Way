import fetch from 'node-fetch';

const SUPABASE_URL = "https://gxpmjqbxtlgkzemdyfwl.supabase.co";
const SERVICE_ROLE_KEY = "TU_SERVICE_ROLE_KEY"; // ← ponla aquí
const FUNCTION_URL = "https://gxpmjqbxtlgkzemdyfwl.functions.supabase.co/send-confirmation-email";

async function main() {
  try {
    const res = await fetch(`${SUPABASE_URL}/rest/v1/leads`, {
      headers: {
        apikey: SERVICE_ROLE_KEY,
        Authorization: `Bearer ${SERVICE_ROLE_KEY}`,
        Prefer: "return=representation",
      },
    });

    const leads = await res.json();

    console.log(`🔍 Total correos encontrados: ${leads.length}`);

    for (const lead of leads) {
      const email = lead.email;

      if (email && email.includes('@')) {
        console.log(`✉️ Enviando a: ${email}`);

        const response = await fetch(FUNCTION_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ email }),
        });

        const data = await response.json();

        if (response.ok) {
          console.log(`✅ Enviado a ${email}`);
        } else {
          console.error(`❌ Error con ${email}:`, data);
        }

        await new Promise(resolve => setTimeout(resolve, 500)); // para no saturar
      }
    }
  } catch (err) {
    console.error("🔥 Error general:", err);
  }
}

main();