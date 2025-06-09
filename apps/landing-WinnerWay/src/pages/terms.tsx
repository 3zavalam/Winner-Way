import React from 'react';
import Footer from '../components/Footer';

const Terms: React.FC = () => {
  return (
    <div className="min-h-screen bg-winner-beige">
      {/* Hero Section */}
      <section className="section pt-10 md:pt-16">
        <div className="winner-container">
          <div className="text-center animate-fade-in">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-winner-green mb-6">
              Terms of Service
            </h1>
            <p className="text-lg md:text-xl text-winner-green/80 mb-8">
              Clear terms for using our tennis analysis platform.
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
                Acceptance of Terms
              </h2>
              <p className="text-lg text-winner-green/80">
                By accessing and using Winner Way's tennis analysis platform, you accept and agree 
                to be bound by the terms and provision of this agreement.
              </p>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-4">
                Service Description
              </h2>
              <p className="text-lg text-winner-green/80">
                Winner Way provides AI-powered tennis video analysis services. Our platform analyzes 
                your tennis videos to provide insights on technique, performance metrics, and 
                improvement recommendations.
              </p>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-6">
                User Responsibilities
              </h2>
              
              <div className="mb-6">
                <h3 className="text-xl font-semibold text-winner-green mb-3">
                  Account Security
                </h3>
                <ul className="space-y-2 text-lg text-winner-green/80">
                  <li className="flex items-center">
                    <span className="text-winner-green mr-2">•</span>
                    Maintain the confidentiality of your account credentials
                  </li>
                  <li className="flex items-center">
                    <span className="text-winner-green mr-2">•</span>
                    Notify us immediately of any unauthorized use
                  </li>
                  <li className="flex items-center">
                    <span className="text-winner-green mr-2">•</span>
                    Use accurate and complete registration information
                  </li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-winner-green mb-3">
                  Content Upload
                </h3>
                <ul className="space-y-2 text-lg text-winner-green/80">
                  <li className="flex items-center">
                    <span className="text-winner-green mr-2">•</span>
                    Only upload videos you own or have permission to use
                  </li>
                  <li className="flex items-center">
                    <span className="text-winner-green mr-2">•</span>
                    Ensure videos contain appropriate content
                  </li>
                  <li className="flex items-center">
                    <span className="text-winner-green mr-2">•</span>
                    Do not upload copyrighted material without permission
                  </li>
                </ul>
              </div>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-4">
                Service Limitations
              </h2>
              <ul className="space-y-2 text-lg text-winner-green/80">
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Analysis results are for informational purposes only
                </li>
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  We do not guarantee specific performance improvements
                </li>
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Service availability may vary
                </li>
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  We reserve the right to modify or discontinue features
                </li>
              </ul>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-4">
                Intellectual Property
              </h2>
              <p className="text-lg text-winner-green/80">
                Winner Way retains all rights to our AI models, algorithms, and platform technology. 
                You retain ownership of your uploaded videos, but grant us license to process them 
                for analysis purposes.
              </p>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-4">
                Payment and Billing
              </h2>
              <ul className="space-y-2 text-lg text-winner-green/80">
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Subscription fees are billed in advance
                </li>
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Refunds are subject to our refund policy
                </li>
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  We may change pricing with advance notice
                </li>
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Failed payments may result in service suspension
                </li>
              </ul>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-4">
                Limitation of Liability
              </h2>
              <p className="text-lg text-winner-green/80">
                Winner Way shall not be liable for any indirect, incidental, special, consequential, 
                or punitive damages resulting from your use of our service.
              </p>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-4">
                Termination
              </h2>
              <p className="text-lg text-winner-green/80">
                We may terminate or suspend your account at any time for violations of these terms. 
                You may cancel your account at any time through your account settings.
              </p>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10 text-center">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-4">
                Contact Information
              </h2>
              <p className="text-lg text-winner-green/80 mb-6">
                For questions about these Terms of Service, contact us:
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

export default Terms; 