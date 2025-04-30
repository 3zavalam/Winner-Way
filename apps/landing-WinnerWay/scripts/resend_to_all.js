import fetch from 'node-fetch';

const SUPABASE_URL = "https://gxpmjqbxtlgkzemdyfwl.supabase.co";
const SERVICE_ROLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd4cG1qcWJ4dGxna3plbWR5ZndsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NTg2MjYyNSwiZXhwIjoyMDYxNDM4NjI1fQ.Wo36k52TPRF4vtJecfEVL5qw089KPi894Rdj5xg5R2k";
const FUNCTION_URL = "https://gxpmjqbxtlgkzemdyfwl.functions.supabase.co/send-confirmation-email";

async function main() {
  try {
    // 1. Obtener todos los leads
    const leadsRes = await fetch(`${SUPABASE_URL}/rest/v1/leads?select=email`, {
      headers: {
        apikey: SERVICE_ROLE_KEY,
        Authorization: `Bearer ${SERVICE_ROLE_KEY}`,
      },
    });

    const leads = await leadsRes.json();
    const leadEmails = leads.map((l) => l.email);

    // 2. Obtener todos los que ya recibieron
    const logsRes = await fetch(`${SUPABASE_URL}/rest/v1/email_logs?select=email`, {
      headers: {
        apikey: SERVICE_ROLE_KEY,
        Authorization: `Bearer ${SERVICE_ROLE_KEY}`,
      },
    });

    const logs = await logsRes.json();
    const sentEmails = new Set(logs.map((l) => l.email));

    // 3. Filtrar los que no han recibido aún
    const toSend = leadEmails.filter((email) => !sentEmails.has(email));

    console.log(`🔍 Correos pendientes de envío: ${toSend.length}`);

    // 4. Enviar correos
    for (const email of toSend) {
      console.log(`✉️ Enviando a: ${email}`);

      const response = await fetch(FUNCTION_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email }),
      });

      const result = await response.json();

      if (response.ok) {
        console.log(`✅ Enviado a ${email}`);
      } else {
        console.error(`❌ Error al enviar a ${email}:`, result);
      }

      // Pequeña pausa para no saturar
      await new Promise((r) => setTimeout(r, 500));
    }
  } catch (err) {
    console.error("🔥 Error general:", err);
  }
}

main();