import React from 'react';
import Logo from './Logo';

const Header: React.FC = () => {
  const scrollToUpload = () => {
    const uploadSection = document.getElementById('upload-section');
    if (uploadSection) {
      uploadSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const scrollToPricing = () => {
    const pricingSection = document.getElementById('buy-now-section');
    if (pricingSection) {
      pricingSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header className="py-6">
      <div className="winner-container flex justify-between items-center">
        <Logo />
        <nav className="hidden md:flex gap-6 items-center">
          <a href="#how-it-works" className="font-medium text-winner-green/90 hover:text-winner-green">How It Works</a>
          <button onClick={scrollToPricing} className="font-medium text-winner-green/90 hover:text-winner-green">Pricing</button>
          <button onClick={scrollToUpload} className="btn-primary">Get Started</button>
        </nav>
        <button className="md:hidden btn-secondary py-2 px-4">Menu</button>
      </div>
    </header>
  );
};

export default Header;
