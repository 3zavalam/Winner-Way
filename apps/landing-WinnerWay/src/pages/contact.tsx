import React from 'react';
import Footer from '../components/Footer';

const Contact: React.FC = () => {
  return (
    <div className="min-h-screen bg-winner-beige">
      {/* Hero Section */}
      <section className="section pt-10 md:pt-16">
        <div className="winner-container">
          <div className="text-center animate-fade-in">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-winner-green mb-6">
              Contact Us
            </h1>
            <p className="text-lg md:text-xl text-winner-green/80 mb-8">
              If you have questions, feedback, or just want to say hi — we'd love to hear from you.
            </p>
            <p className="text-sm text-winner-green/60 mb-8">
              We try to reply within 24–48 hours.
            </p>
          </div>
        </div>
      </section>

      {/* Content */}
      <section className="section">
        <div className="winner-container max-w-4xl">
          <div className="animate-fade-in space-y-12">
            
            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10 text-center">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-6">
                📩 Email
              </h2>
              <p className="text-lg text-winner-green/80 mb-6">
                For general inquiries, support, or feedback:
              </p>
              <a 
                href="mailto:winnerwayai@gmail.com" 
                className="btn-primary inline-block text-lg"
              >
                winnerwayai@gmail.com
              </a>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-6">
                🔗 Follow Us
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="text-center">
                  <h3 className="text-xl font-semibold text-winner-green mb-3">
                    Instagram
                  </h3>
                  <a 
                    href="https://instagram.com/winnerwayai" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="btn-secondary inline-block"
                  >
                    @winnerwayai
                  </a>
                </div>
                <div className="text-center">
                  <h3 className="text-xl font-semibold text-winner-green mb-3">
                    TikTok
                  </h3>
                  <a 
                    href="https://tiktok.com/@winnerwayai" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="btn-secondary inline-block"
                  >
                    @winnerwayai
                  </a>
                </div>
              </div>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-6">
                📧 Newsletter & Updates
              </h2>
              <div className="space-y-4">
                <p className="text-lg text-winner-green/80">
                  Want to unsubscribe from our emails or manage your communication preferences?
                </p>
                <ul className="space-y-2 text-lg text-winner-green/80">
                  <li className="flex items-center">
                    <span className="text-winner-green mr-2">•</span>
                    Send us an email at{' '}
                    <a href="mailto:winnerwayai@gmail.com" className="text-winner-green hover:underline mx-1">
                      winnerwayai@gmail.com
                    </a>
                  </li>
                  <li className="flex items-center">
                    <span className="text-winner-green mr-2">•</span>
                    Use the unsubscribe link in any of our emails
                  </li>
                  <li className="flex items-center">
                    <span className="text-winner-green mr-2">•</span>
                    Contact us to update your email preferences
                  </li>
                </ul>
              </div>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-6">
                💡 Feedback & Suggestions
              </h2>
              <p className="text-lg text-winner-green/80 mb-4">
                We're always looking to improve Winner Way. Your feedback helps us build better features for tennis players everywhere.
              </p>
              <ul className="space-y-2 text-lg text-winner-green/80">
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Found a bug? Let us know!
                </li>
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Have an idea for a new feature?
                </li>
                <li className="flex items-center">
                  <span className="text-winner-green mr-2">•</span>
                  Want to share your success story?
                </li>
              </ul>
            </div>

            <div className="bg-white/50 rounded-lg p-8 border border-winner-green/10">
              <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-6">
                🏢 Business Inquiries
              </h2>
              <p className="text-lg text-winner-green/80 mb-4">
                For partnerships, press inquiries, or business opportunities:
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

export default Contact; 