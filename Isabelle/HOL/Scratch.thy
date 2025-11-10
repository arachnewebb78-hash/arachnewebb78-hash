theory Scratch
  imports Main
begin

datatype nat = Zero | Suc nat

fun add :: "nat \<Rightarrow> nat \<Rightarrow> nat" where
  "add Zero  n = n" |
  "add (Suc m) n = Suc (add m n)"

(* Helper lemma: add to right-hand Suc *)
lemma add_suc_right: "add a (Suc b) = Suc (add a b)"
  by (induction a) auto

lemma add_zero_right: "add n Zero = n"
  by (induction n) auto

declare add_zero_right [simp]

declare add_suc_right [simp]

(* Main lemma: commutativity of addition *)
lemma add_comm: "add a b = add b a"
  apply (induction a)
   apply auto
  done

end
