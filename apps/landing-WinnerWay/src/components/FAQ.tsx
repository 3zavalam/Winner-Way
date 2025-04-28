
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

const faqItems = [
  {
    question: "When will the beta be available?",
    answer: "We're planning to launch our beta program in Summer 2025. Early sign-ups will receive priority access."
  },
  {
    question: "Is Winner Way free to use?",
    answer: "The beta version will be free for early adopters. After the beta period, we'll offer both free and premium subscription options."
  },
  {
    question: "What devices is Winner Way compatible with?",
    answer: "Winner Way works with iOS and Android devices. You'll need a smartphone with a decent camera to record your tennis swings."
  },
  {
    question: "How accurate is the AI analysis?",
    answer: "Our AI has been trained on thousands of professional tennis swings and provides detailed analysis with up to 95% accuracy when compared to professional coaching assessments."
  },
  {
    question: "Can I use Winner Way for other sports?",
    answer: "Currently, Winner Way is specifically designed for tennis. We may expand to other sports in the future based on user demand."
  }
];

const FAQ = () => {
  return (
    <section id="faq" className="py-16 px-4">
      <div className="max-w-3xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold text-winnerGreen text-center mb-12">
          Frequently Asked Questions
        </h2>
        
        <Accordion type="single" collapsible className="w-full">
          {faqItems.map((item, index) => (
            <AccordionItem key={index} value={`item-${index}`} className="border-winnerGreen/20">
              <AccordionTrigger className="text-winnerGreen text-lg font-medium hover:text-winnerGreen/80">
                {item.question}
              </AccordionTrigger>
              <AccordionContent className="text-winnerGreen/80">
                {item.answer}
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </div>
    </section>
  );
};

export default FAQ;
