use crate::polyfit::Polynomial;

struct Measurement<E> {
    value: E,
    uncertainty: E,
}

impl<E> Measurement<E> {
    fn compute_unknown(&self, fit: &Polynomial<E>) -> Self {
        todo!()

    }
}

