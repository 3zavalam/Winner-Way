import { createClient } from '@supabase/supabase-js';
import type { Database } from './types';

//const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
//const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;

const SUPABASE_URL = "https://gxpmjqbxtlgkzemdyfwl.supabase.co"
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd4cG1qcWJ4dGxna3plbWR5ZndsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU4NjI2MjUsImV4cCI6MjA2MTQzODYyNX0.di53Dy6RiAP5quYQ8OvpKn9LF6CAdTsOwKL0CKsiRYU"

export const supabase = createClient<Database>(SUPABASE_URL, SUPABASE_ANON_KEY);