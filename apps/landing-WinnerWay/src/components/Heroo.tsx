
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
        // Handle unique constraint violation
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
    <section className="py-16 md:py-24 px-4 text-center max-w-4xl mx-auto">
      <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-winnerGreen mb-6 animate-fade-in">
        Join the Winner Way Beta
      </h1>
      
      <p className="text-lg md:text-xl mb-10 text-winnerGreen/80 max-w-2xl mx-auto">
        Imagine your best game — every shot smarter, every move sharper. Be one of the first to train with AI-driven coaching.
      </p>
      
      <form onSubmit={handleSubmit} className="flex flex-col md:flex-row gap-3 max-w-md mx-auto mb-12">
        <Input
          type="email"
          placeholder="Your email address"
          className="bg-white border-winnerGreen/30 focus:border-winnerGreen h-12"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <Button 
          type="submit" 
          className="bg-winnerGreen text-white hover:bg-winnerGreen/90 h-12 px-6"
          disabled={isLoading}
        >
          {isLoading ? "Joining..." : "Join the Beta!"}
        </Button>
      </form>
      
      <div className="max-w-2xl mx-auto px-4 py-5 bg-white rounded-lg shadow-sm">
        <p className="text-winnerGreen/90">
          Winner Way was born out of a simple idea: that every player deserves coaching they can trust.
          We're players too — and we're building this tool to help you unlock your full potential.
        </p>
      </div>
    </section>
  );
};

export default Hero;
