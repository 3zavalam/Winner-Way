
import Header from "@/components/Header";
import Hero from "@/components/Hero";
import HowItWorks from "@/components/HowItWorks";
import CaseStudy from "@/components/CaseStudy";
import Benefits from "@/components/Benefits";
import FAQ from "@/components/FAQ";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-winnerBeige">
      <Header />
      <Hero />
      <HowItWorks />
      <CaseStudy />
      <Benefits />
      <FAQ />
      <Footer />
    </div>
  );
};

export default Index;
