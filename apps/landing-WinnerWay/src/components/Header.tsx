
import { useState } from "react";
import { Button } from "./ui/button";
import { ChevronDown, Menu } from "lucide-react";

const Header = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
    setMobileMenuOpen(false);
  };

  return (
    <header className="py-4 px-4 md:px-8 flex items-center justify-between relative z-10">
      <div className="flex items-center">
        <img
          src="/lovable-uploads/410138d7-1bd3-4e47-8171-94b790ef10e5.png"
          alt="Winner Way Logo"
          className="h-10 md:h-12"
        />
        <span className="ml-3 text-winnerGreen font-bold text-xl md:text-2xl">Winner Way</span>
      </div>
      
      {/* Mobile menu button */}
      <div className="md:hidden">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label="Toggle menu"
        >
          <Menu className="h-6 w-6 text-winnerGreen" />
        </Button>
      </div>

      {/* Mobile menu */}
      {mobileMenuOpen && (
        <div className="absolute top-full right-0 left-0 bg-winnerBeige py-4 px-4 shadow-md md:hidden animate-fade-in">
          <nav className="flex flex-col space-y-3">
            <button
              onClick={() => scrollToSection('how-it-works')}
              className="text-winnerGreen hover:text-opacity-80 text-left py-2 border-b border-winnerGreen/20"
            >
              How It Works
            </button>
            <button
              onClick={() => scrollToSection('benefits')}
              className="text-winnerGreen hover:text-opacity-80 text-left py-2 border-b border-winnerGreen/20"
            >
              Benefits
            </button>
            <button
              onClick={() => scrollToSection('faq')}
              className="text-winnerGreen hover:text-opacity-80 text-left py-2"
            >
              FAQ
            </button>
          </nav>
        </div>
      )}

      {/* Desktop navigation */}
      <nav className="hidden md:flex items-center space-x-8">
        <button
          onClick={() => scrollToSection('how-it-works')}
          className="text-winnerGreen hover:text-opacity-80 font-medium"
        >
          How It Works
        </button>
        <button
          onClick={() => scrollToSection('benefits')}
          className="text-winnerGreen hover:text-opacity-80 font-medium"
        >
          Benefits
        </button>
        <button
          onClick={() => scrollToSection('faq')}
          className="text-winnerGreen hover:text-opacity-80 font-medium"
        >
          FAQ
        </button>
      </nav>
    </header>
  );
};

export default Header;
