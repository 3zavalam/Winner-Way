import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="py-8 bg-white/30">
      <div className="winner-container">
        <div className="text-center">
          <p className="text-winner-green/70 mb-4">
            Built by WinnerWay. Powered by Python & AI.
          </p>
          <div className="flex justify-center gap-6">
            <a 
              href="/privacy.html" 
              className="text-winner-green/70 hover:text-winner-green text-sm"
            >
              Privacy Policy
            </a>
            <a 
              href="/terms.html" 
              className="text-winner-green/70 hover:text-winner-green text-sm"
            >
              Terms of Service
            </a>
            <a 
              href="/contact" 
              className="text-winner-green/70 hover:text-winner-green text-sm"
            >
              Contact
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;