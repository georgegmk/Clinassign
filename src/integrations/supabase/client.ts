// This file is automatically generated. Do not edit it directly.
import { createClient } from '@supabase/supabase-js';
import type { Database } from './types';

const SUPABASE_URL = "https://acmihgikuqzekgewttyf.supabase.co";
const SUPABASE_PUBLISHABLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFjbWloZ2lrdXF6ZWtnZXd0dHlmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA3NTA2MjAsImV4cCI6MjA1NjMyNjYyMH0.OXknSZFFSjy7Q3SnjDeYIfgBJ25l8MTkxm39OJE2xoM";

// Import the supabase client like this:
// import { supabase } from "@/integrations/supabase/client";

export const supabase = createClient<Database>(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY);