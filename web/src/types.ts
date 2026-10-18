export type Mode = 'fixture' | 'live_phone' | 'live_glasses' | 'recorded_glasses';
export type Recipient = 'self' | 'household';
export type Nutrition = {
  basis_grams: number;
  carbohydrate_g: number | null;
  fiber_g: number | null;
  sugars_g: number | null;
  protein_g: number | null;
  sodium_mg: number | null;
};
export type Product = {
  id: string; name: string; brand?: string | null; barcode?: string | null; market: string;
  ingredients: string[]; nutrition: Nutrition;
  provenance: { kind: 'synthetic' | 'package_ocr' | 'reviewed_label' | 'database'; source: string; captured_at: string; verified: boolean };
  uncertain_fields: string[];
};
export type GoalSource = { kind: 'self_selected' | 'reviewed_guidance' | 'synthetic'; source_date: string | null; confirmed: true; reviewed_at?: string };
export type Profile = { display_name: string; goals: string[]; avoid_ingredients: string[]; consented: boolean; goal_source?: GoalSource | null };
export type CartItem = { id: string; product_id: string; product_name: string; recipient_id: string; quantity: number; confirmed_at: string };
export type Action = { id: string; label: string; product_id: string; recipient_id: string; quantity: number; expires_at: string };
export type Message = { id: string; role: 'user' | 'assistant'; text: string; created_at: string };
export type Session = {
  id: string; mode: Mode; recipient_id: Recipient; portion_grams: number | null; generation: number;
  profile: Profile; products: Product[]; current_product_id: string | null; cart: CartItem[];
  pending_action: Action | null; messages: Message[]; created_at: string;
};
export type Evidence = { label: string; value: string; source: string; kind: Product['provenance']['kind'] | 'calculation'; verified: boolean };
export type Comparison = {
  portion_grams: number; recipient_id: string;
  items: ({ product_id: string; name: string } & Omit<Nutrition, 'basis_grams'>)[];
  summary: string;
};
export type EvidenceEvent = { id: string; type: string; at: string; stage: string; message: string; media_time_s?: number | null; box?: { x: number; y: number; w: number; h: number } | null };
export type TurnResponse = {
  session: Session; reply: string;
  status: 'ready' | 'needs_image' | 'needs_portion' | 'needs_review' | 'needs_confirmation' | 'unknown' | 'cancelled';
  evidence: Evidence[]; comparison: Comparison | null; events: EvidenceEvent[]; pending_action: Action | null;
};
export type Health = { status: string; providers: { perceptron: boolean; gemini: boolean }; version: string };
export type TurnRequest = {
  text: string; client_turn_id: string; recipient_id?: Recipient; portion_grams?: number;
  fixture_product_id?: string; image_data_url?: string; captured_at?: string; media_time_s?: number;
};
export type VoiceRecognitionResult = { isFinal: boolean; 0: { transcript: string } };
export interface VoiceRecognition {
  lang: string; continuous: boolean; interimResults: boolean;
  onresult: ((event: { results: ArrayLike<VoiceRecognitionResult> }) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start(): void; stop(): void; abort(): void;
}
declare global {
  interface Window { SpeechRecognition?: new () => VoiceRecognition; webkitSpeechRecognition?: new () => VoiceRecognition }
}
