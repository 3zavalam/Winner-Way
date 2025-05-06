import { useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { toast } from "./ui/use-toast";
import { supabase } from "@/integrations/supabase/client";

const Hero = () => {
  const [email, setEmail] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    if (!email || !email.includes('@')) {
      toast({
        title: "Please enter a valid email",
        description: "We need your email to send you beta access.",
        variant: "destructive",
      });
      setIsLoading(false);
      return;
    }

    try {
      const { error } = await supabase
        .from("leads")
        .insert([{ email, source: "beta-landing" }]);

      if (error) {
        if (error.code === '23505') {
          toast({
            title: "You're already on our list!",
            description: "This email is already registered for early access.",
            duration: 5000,
          });
        } else {
          console.error("Error saving email:", error);
          toast({
            title: "Something went wrong",
            description: "Please try again later.",
            variant: "destructive",
          });
        }
      } else {
        try {
          await fetch("https://gxpmjqbxtlgkzemdyfwl.functions.supabase.co/send-confirmation-email", {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify({ email }),
          });
        } catch (error) {
          console.error("Failed to send confirmation email", error);
        }

        toast({
          title: "Success!",
          description: "You're on the list for early access!",
          duration: 5000,
        });
        setEmail("");
      }
    } catch (error) {
      console.error("Exception:", error);
      toast({
        title: "Something went wrong",
        description: "Please try again later.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section className="py-16 md:py-24 px-4 max-w-7xl mx-auto relative flex flex-col lg:flex-row items-center justify-between gap-12">
      {/* Columna izquierda: texto y formulario */}
      <div className="text-center lg:text-left flex-[3]">
        <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-winnerGreen mb-6 animate-fade-in">
          It’s Like Having a Tennis Coach in Your Pocket
        </h1>
        <p className="text-winnerGreen/80 text-lg md:text-xl mb-10 max-w-2xl">
          Winner Way helps you spot your mistakes, track your progress, and get tips tailored to your level — so you stop guessing and start improving.
        </p>

        <div className="flex flex-col md:flex-row items-center justify-center lg:justify-start gap-4 mb-6">
          <form onSubmit={handleSubmit} className="flex flex-col md:flex-row gap-4 w-full max-w-xl">
            <Input
              type="email"
              placeholder="Your email address"
              className="flex-1 bg-white border-winnerGreen/30 focus:border-winnerGreen focus:ring-2 focus:ring-winnerGreen h-14 text-lg px-6"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
            <Button 
              type="submit" 
              className="bg-winnerGreen text-white hover:brightness-110 hover:shadow-lg transition-all h-14 px-8 text-lg font-bold"
              disabled={isLoading}
            >
              {isLoading ? "Joining..." : "🎾 Get My AI Training Tips"}
            </Button>
          </form>
        </div>

        <p className="text-winnerGreen/70 text-sm">
          We respect your privacy. No spam, just smarter tennis.
        </p>
      </div>

      {/* Columna derecha: mascota */}
      <div className="flex-[1] flex justify-center">
        <img
          src="/mascot.png"
          alt="Mascot"
          className="w-[300px] h-auto opacity-90 pointer-events-none select-none"
        />
      </div>
    </section>
  );
};

export default Hero;