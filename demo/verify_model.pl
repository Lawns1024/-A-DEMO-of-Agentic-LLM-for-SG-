:- initialization(main, main).

valid_probability(P) :-
    number(P),
    P >= 0.0,
    P =< 1.0.

valid_transition(State, Action) :-
    findall(P, transition(State, Action, _, P), Probabilities),
    expected_state_count(ExpectedStates),
    length(Probabilities, ExpectedStates),
    maplist(valid_probability, Probabilities),
    sum_list(Probabilities, Total),
    Difference is abs(Total - 1.0),
    Difference < 1.0e-9.

valid_reward(State, Action) :-
    findall(R, reward(State, Action, _, R), Rewards),
    expected_agent_count(ExpectedAgents),
    length(Rewards, ExpectedAgents),
    maplist(number, Rewards).

valid_profile(Action) :-
    profile(Action, MinMultiplier, MaxMultiplier),
    integer(MinMultiplier),
    integer(MaxMultiplier),
    MinMultiplier > 0,
    MaxMultiplier >= MinMultiplier.

verify_counts :-
    setof(State, Action^Next^P^transition(State, Action, Next, P), States),
    expected_state_count(ExpectedStates),
    length(States, ExpectedStates),
    forall(
        member(State, States),
        (
            setof(Action, Next^P^transition(State, Action, Next, P), Actions),
            expected_action_count(ExpectedActions),
            length(Actions, ExpectedActions)
        )
    ).

verify_all :-
    verify_counts,
    forall(transition(State, Action, _, _), valid_transition(State, Action)),
    forall(reward(State, Action, _, _), valid_reward(State, Action)),
    forall(profile(Action, _, _), valid_profile(Action)).

main([FactsPath]) :-
    consult(FactsPath),
    ( verify_all ->
        writeln('Prolog verification passed.'),
        halt(0)
    ;
        writeln(user_error, 'Prolog verification failed.'),
        halt(1)
    ).
main(_) :-
    writeln(user_error, 'Usage: swipl -q -s verify_model.pl -- FACTS_FILE'),
    halt(2).
