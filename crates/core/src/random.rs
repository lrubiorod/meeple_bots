/// Small RNG interface passed explicitly to agents for deterministic simulations.
pub trait RandomSource {
    fn next_u64(&mut self) -> u64;

    fn index(&mut self, upper_bound: usize) -> Option<usize> {
        if upper_bound == 0 {
            return None;
        }

        let upper_bound = upper_bound as u64;
        let rejection_threshold = upper_bound.wrapping_neg() % upper_bound;
        loop {
            let candidate = self.next_u64();
            if candidate >= rejection_threshold {
                return Some((candidate % upper_bound) as usize);
            }
        }
    }

    fn unit_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }
}
