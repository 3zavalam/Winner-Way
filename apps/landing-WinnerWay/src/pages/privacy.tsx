import React from 'react';
import Footer from '../components/Footer';

const Privacy: React.FC = () => {
  return (
    <div className="min-h-screen bg-winner-beige">
      {/* Hero Section */}
      <section className="section pt-10 md:pt-16">
        <div className="winner-container">
          <div className="text-center animate-fade-in">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-winner-green mb-6">
              Privacy Policy
            </h1>
            <p className="text-lg md:text-xl text-winner-green/80 mb-8">
              Your privacy matters to us. Here's how we protect your data.
            </p>
            <p className="text-sm text-winner-green/60 mb-8">
              Last updated: January 2025
            </p>
          </div>
        </div>
      </section>

      {/* Content */}
      <section className="section">
        <div className="winner-container max-w-4xl">
          <div className="animate-fade-in space-y-12">
            
            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-4">
                Introduction
              </h2>
              <p className="text-lg text-winner-green/80">
                Winner Way we is committed to protecting your privacy. 
                This Privacy Policy explains how we collect, use, disclose, and safeguard your 
                information when you use our tennis analysis platform.
              </p>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-6">
                Information We Collect
              </h2>
              
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-winner-green mb-3">
                  Personal Information
                </h3>
                <ul className="space-y-2 text-lg text-winner-green/80">
                  <li className="flex items-center">
                    <span className="text-winner-green mr-2">•</span>
                    Email address
                  </li>
                  <li className="flex items-center">
                    <span className="text-winner-green mr-2">•</span>
                    Name and profile information
                  </li>
                  <li className="flex items-center">
                    <span className="text-winner-green mr-2">•</span>
                    Account credentials
                  </li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-winner-green mb-3">
                  Tennis Data
                </h3>
                <ul className="space-y-2 text-lg text-winner-green/80">
                  <li className="flex items-center">
                    <span className="text-winner-green mr-2">•</span>
                    Uploaded tennis videos
                  </li>
                  <li className="flex items-center">
                    <span className="text-winner-green mr-2">•</span>
                    Analysis results and metrics
                  </li>
                  <li className="flex items-center">
                    <span className="text-winner-green mr-2">•</span>
                    Performance data and statistics
                  </li>
                </ul>
              </div>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-4">
                How We Use Your Information
              </h2>
              <ul className="space-y-2 text-lg text-winner-green/80">
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Provide and improve our tennis analysis services
                </li>
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Process and analyze your tennis videos
                </li>
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Send important updates about your account
                </li>
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Provide customer support
                </li>
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Improve our AI models and algorithms
                </li>
              </ul>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-4">
                Data Security
              </h2>
              <p className="text-lg text-winner-green/80">
                We implement appropriate security measures to protect your personal information 
                against unauthorized access, alteration, disclosure, or destruction. Your data 
                is encrypted and stored securely.
              </p>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-4">
                Your Rights
              </h2>
              <p className="text-lg text-winner-green/80 mb-4">You have the right to:</p>
              <ul className="space-y-2 text-lg text-winner-green/80">
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Access your personal data
                </li>
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Correct inaccurate data
                </li>
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Delete your account and data
                </li>
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Export your data
                </li>
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Opt out of communications
                </li>
              </ul>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10 text-center">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-4">
                Contact Us
              </h2>
              <p className="text-lg text-winner-green/80 mb-6">
                If you have questions about this Privacy Policy, contact us:
              </p>
              <a 
                href="mailto:winnerwayai@gmail.com" 
                className="btn-primary inline-block"
              >
                winnerwayai@gmail.com
              </a>
            </div>

          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Privacy; 