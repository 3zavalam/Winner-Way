import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="py-10 bg-winner-beige border-t border-winner-green/10">
      <div className="winner-container text-center">
        <p className="text-sm text-winner-green/70 mb-4">
          Built by WinnerWay. Powered by Python & AI.
        </p>
        <div className="flex flex-wrap justify-center gap-6 text-sm">
          <a
            href="/privacy"
            className="text-winner-green/70 hover:text-winner-green transition-colors"
          >
            Privacy Policy
          </a>
          <a
            href="/terms"
            className="text-winner-green/70 hover:text-winner-green transition-colors"
          >
            Terms of Service
          </a>
          <a
            href="/contact"
            className="text-winner-green/70 hover:text-winner-green transition-colors"
          >
            Contact
          </a>
        </div>
        <p className="mt-6 text-xs text-muted-foreground">
          &copy; 2025 Winner Way. All rights reserved.
        </p>
      </div>
    </footer>
  );
};

export default Footer;